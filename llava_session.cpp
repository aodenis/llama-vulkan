#include "llava_session.h"
#include "llava_layer_session_data.h"
#include "llava_context.h"
#include "llava_buffer.h"
#include "llava_device_memory.h"
#include "utils.h"
#include "llava_command_buffer.h"
#include <chrono>
#include "ggml_file.h"
#include <cmath>
#include <iostream>
#include <set>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#ifdef RUNTIME_BUILD_ENABLED
#include <glslang/Public/ShaderLang.h>
#include <random>
#endif

const u32 min_backlog_size = 128;
const u32 max_backlog_size = 2048;

llava_session::llava_session(llava_context* _ctx) : rng(time(nullptr)), mirostat_mu(2 * mirostat_tau), ctx(_ctx), model(_ctx->get_model()), backlog_size(min_backlog_size) { // NOLINT(cert-msc51-cpp)

}

llava_session::~llava_session() {
    for (auto& x : layer_data) {
        delete x;
    }
    delete command_buffer;
    command_buffer = nullptr;
    reset_main_buffers();
}

void llava_session::ensure_buffers_created() {
    if (not layer_data.empty()) {
        return;
    }
    layer_data.reserve(ctx->layers.size());
    for (u32 i = 0; i < ctx->layers.size(); ++i) {
        layer_data.push_back(new llava_layer_session_data(this));
    }
}

void llava_session::recreate_buffers() {
    reset_main_buffers();
    if (batch_size == 0) {
        return;
    }

    u32 dim = model->header.dim;
    u32 ff_size = model->ff_size;
    u32 n_heads =  model->header.n_heads;
    u32 vocab_size = model->header.vocab_size;

    // Create main buffers
    main_buffer_memory = new llava_device_memory(ctx);
    current_thought = new llava_buffer(ctx, ggml_value_type::f32, dim, batch_size, main_buffer_memory);
    current_thought_sublayer = new llava_buffer(ctx, ggml_value_type::f32, dim, batch_size, main_buffer_memory);
    current_thought_middle_normd = new llava_buffer(ctx, ggml_value_type::f32, dim, batch_size, main_buffer_memory);
    properties_mask = new llava_buffer(ctx, ggml_value_type::f32, ff_size, batch_size, main_buffer_memory);
    main_ff_result = new llava_buffer(ctx, ggml_value_type::f32, ff_size, batch_size, main_buffer_memory);
    current_Q = new llava_buffer(ctx, ggml_value_type::f32, dim, batch_size, main_buffer_memory);
    current_K = new llava_buffer(ctx, ggml_value_type::f32, dim, batch_size, main_buffer_memory);
    main_attn_result = new llava_buffer(ctx, ggml_value_type::f32, backlog_size, n_heads * batch_size, main_buffer_memory);
    config_buffer = new llava_buffer(ctx, ggml_value_type::f32, 4, 1, main_buffer_memory);
    current_V = new llava_buffer(ctx, ggml_value_type::f32, dim, batch_size, main_buffer_memory);
    current_Vout = new llava_buffer(ctx, ggml_value_type::f32, dim, batch_size, main_buffer_memory);
    norm_w = new llava_buffer(ctx, model->get_buffer_descriptor("norm"), main_buffer_memory);
    output_w = new llava_buffer(ctx, model->get_buffer_descriptor("output"), main_buffer_memory);
    output_probs = new llava_buffer(ctx, ggml_value_type::f32, vocab_size, batch_size, main_buffer_memory);
    main_buffer_memory->freeze();
    norm_w->load_to_gpu();
    output_w->load_to_gpu();
}

void llava_session::reset_main_buffers() {
    delete current_thought;
    delete current_thought_sublayer;
    delete current_Q;
    delete current_thought_middle_normd;
    delete current_K;
    delete current_V;
    delete current_Vout;
    delete main_attn_result;
    delete config_buffer;
    delete norm_w;
    delete output_w;
    delete output_probs;
    delete properties_mask;
    delete main_ff_result;
    delete main_buffer_memory;
    current_thought = nullptr;
    current_thought_sublayer = nullptr;
    current_thought_middle_normd = nullptr;
    current_Q = nullptr;
    current_K = nullptr;
    current_V = nullptr;
    current_Vout = nullptr;
    main_attn_result = nullptr;
    config_buffer = nullptr;
    norm_w = nullptr;
    output_w = nullptr;
    output_probs = nullptr;
    properties_mask = nullptr;
    main_ff_result = nullptr;
    main_buffer_memory = nullptr;
}

u32 llava_session::get_last_predicted_token(bool deterministic) {
    // based on llama_sample_token_mirostat_v2 from llama.cpp
    auto n_vocab = model->header.vocab_size;

    map<u32, u32> last_token_count;
    if (not deterministic) {
        for (u32 i = (repeat_last_n > token_buffer.size()) ? 0 : (token_buffer.size() - repeat_last_n); i < token_buffer.size(); ++i) {
            last_token_count[token_buffer.at(i)]++;
        }
    }

    if (command_buffer) {
        command_buffer->wait_idle();
    }

    vector<float> pulled_data;
    pulled_data.resize(model->header.vocab_size);
    {
        auto* res = static_cast<float *>(output_probs->map(0, (batch_size - 1) * model->header.vocab_size * sizeof(float), model->header.vocab_size * sizeof(float)));
        memcpy(pulled_data.data(), res, model->header.vocab_size * sizeof(float));
        output_probs->unmap();
    }

    for (float& x : pulled_data) {
        if (isnan(x) or isinf(x)) {
            cerr << "NaN in output buffer" << endl;
            break;
        }
    }

    if (deterministic) {
        u32 m = 0;
        for (u32 token_id = 0; token_id < n_vocab; token_id++) {
            if (pulled_data[token_id] > pulled_data[m]) {
                m = token_id;
            }
        }
        return m;
    }

    std::vector<unsigned int> back_id;
    back_id.reserve(n_vocab);

    for (auto& [tok_id, count] : last_token_count) {
        if (tok_id != 13) {
            continue; // Skip \n
        }
        if(pulled_data.at(tok_id) <= 0) {
            pulled_data.at(tok_id) *= repeat_penalty;
        } else {
            pulled_data.at(tok_id) /= repeat_penalty;
        }
        pulled_data.at(tok_id) -= alpha_frequency * float(count) + alpha_presence;
    }

    float total = 0;
    for (u32 i = 0; i < n_vocab; ++i) {
        pulled_data.at(i) = expf(pulled_data.at(i) / temp);
        total += pulled_data.at(i);
    }

    u32 j_w = 0;
    float new_sum = 0.;
    float cut = powf(0.5, mirostat_mu);
    for (u32 i = 0; i < n_vocab; ++i) {
        pulled_data.at(i) /= total;
        if (pulled_data.at(i) >= cut) {
            new_sum += pulled_data.at(i);
            if (j_w != i) {
                pulled_data.at(j_w) = pulled_data.at(i);
            }
            back_id.push_back(i);
            j_w++;
        }
    }

    std::discrete_distribution<> dist(pulled_data.begin(), pulled_data.begin() + j_w);
    int idx = dist(rng);

    mirostat_mu += mirostat_eta * (mirostat_tau + log2f(pulled_data.at(idx) / new_sum));

    return back_id.at(idx);
}

bool llava_session::start_next_token_prediction() {
    ensure_buffers_created();
    u32 const to_process = token_buffer.size() - current_tokens_in_gpu;
    if (to_process == 0) {
        return false;
    }
    set_batch_size(to_process);
    if (command_buffer == nullptr) {
        recreate_spevars();
        command_buffer = new llava_command_buffer(this);
        command_buffer->record_execution();
    }

    ggml_data_descriptor const& descriptor = model->get_buffer_descriptor("tok_embeddings");
    assert((descriptor.size % model->tokens.size()) == 0);

    if(token_buffer.size() > backlog_size) {
        cerr << "Token buffer overflow" << endl;
        return false;
    }

    for (u32 i = 0; i < to_process; i++) {
        u32 token_id = token_buffer.at(i + current_tokens_in_gpu);
        assert(token_id < model->tokens.size());
        current_thought->write_f32(model->mapping + (descriptor.offset + token_id * (descriptor.size / model->tokens.size())), descriptor.ftype, descriptor.model_version, i * model->header.dim, model->header.dim);
    }
    u32 config[4] = {current_tokens_in_gpu, current_tokens_in_gpu, current_tokens_in_gpu, current_tokens_in_gpu};
    config_buffer->write_f32(&(config[0]), ggml_value_type::f32, 1, 0, 4);
    command_buffer->run();
    current_tokens_in_gpu += to_process;
    return true;
}

u32 llava_session::finish_next_token_prediction() {
    command_buffer->wait_idle();
    return get_last_predicted_token(false);
}

u32 llava_session::predict_next_token() {
    if (not start_next_token_prediction()) {
        return ~0U;
    }
    return finish_next_token_prediction();
}

void llava_session::set_batch_size(u32 _batch_size) {
    if (batch_size == _batch_size) {
        return;
    }

    delete command_buffer;
    command_buffer = nullptr;
    batch_size = _batch_size;

    recreate_buffers();
}

bool llava_session::set_text(const string& new_text) {
    vector<u32> dec_tokens;
    model->tokenize(dec_tokens, new_text, true);
    u32 next_backlog_size = backlog_size;
    while (dec_tokens.size() > next_backlog_size) {
        next_backlog_size <<= 1;
    }
    if (next_backlog_size > max_backlog_size) {
        return false;
    }
    if (not set_backlog_size(next_backlog_size)) {
        return false;
    }
    u32 max_prefix = 0;
    while ((max_prefix < dec_tokens.size()) and (max_prefix < token_buffer.size()) and (max_prefix < current_tokens_in_gpu) and (token_buffer.at(max_prefix) == new_text.at(max_prefix))) {
        max_prefix++;
    }
    token_buffer = dec_tokens;
    current_tokens_in_gpu = max_prefix;
    return true;
}

bool llava_session::add_text(const string &s) {
    string full_text = model->tokens_to_text(token_buffer.data(), token_buffer.size()) + s;
    return set_text(full_text);
}

bool llava_session::push_token(u32 new_token) {
    assert(new_token < model->tokens.size());
    if (token_buffer.size() == backlog_size) {
        if (backlog_size == max_backlog_size) {
            return false;
        } else {
            if (not set_backlog_size(backlog_size * 2)) {
                return false;
            }
        }
    }
    token_buffer.push_back(new_token);
    return true;
}

vector<llava_layer_session_data *> const &llava_session::get_layer_data() const {
    return layer_data;
}

bool llava_session::set_backlog_size(u32 new_size) {
    if(new_size > max_backlog_size) {
        return false;
    }
    if (new_size == backlog_size) {
        return true;
    }
    if (new_size < min_backlog_size) {
        return false;
    }
    if (new_size < current_tokens_in_gpu) {
        return false;
    }

    delete command_buffer;
    command_buffer = nullptr;

    backlog_size = new_size;
    recreate_buffers();

    flush_layers_data_buffers();

    recreate_spevars();
    return true;
}

void llava_session::flush_layers_data_buffers() {
    for(llava_layer_session_data* l_data : layer_data) {
        l_data->flush_buffers_on_gpu();
    }
}

specialization_variables_t const &llava_session::get_spevar_struct() const {
    return specialization_variables;
}

void llava_session::recreate_spevars() {
    auto& spevar = specialization_variables;

    // Compute specialization variables
    u32 dim = model->header.dim;
    u32 ff_size = model->ff_size;
    u32 n_heads =  model->header.n_heads;
    u32 rot = model->header.rot;
    u32 workgroup_size = ctx->workgroup_size;

    spevar.head_count = n_heads;
    spevar.rot = rot;
    spevar.quarterrot = rot / 4;
    spevar.rot_bits = ulog2(rot);
    spevar.max_wgs = workgroup_size;
    spevar.max_wgs_bits = ulog2(workgroup_size);
    spevar.ff_size = ff_size;

    if (dim == 4096) {
        spevar.matmul_dim_row_worker_count = 128;
    } else if (dim == 5120) {
        spevar.matmul_dim_row_worker_count = 32;
    } else {
        assert(false);
    }

    assert((workgroup_size % spevar.matmul_dim_row_worker_count) == 0);
    spevar.matmul_dim_row_per_wavefront = workgroup_size / spevar.matmul_dim_row_worker_count;
    spevar.matmul_dim_row_worker_count_log2 = ulog2(spevar.matmul_dim_row_worker_count);
    spevar.matmul_dim_q4_blocks_per_row = dim / 32;
    spevar.matmul_dim_q4_block_count_per_worker = updiv(spevar.matmul_dim_q4_blocks_per_row, spevar.matmul_dim_row_worker_count);

    if ((ff_size == 11008) or (ff_size == 13824)) {
        spevar.matmul_ff_row_worker_count = 128;
    } else {
        assert(false);
    }

    assert((workgroup_size % spevar.matmul_ff_row_worker_count) == 0);

    spevar.matmul_ff_row_per_wavefront = workgroup_size / spevar.matmul_ff_row_worker_count;
    spevar.matmul_ff_row_worker_count_log2 = ulog2(spevar.matmul_ff_row_worker_count);
    spevar.matmul_ff_q4_blocks_per_row = ff_size / 32;
    spevar.matmul_ff_q4_block_count_per_worker = updiv(spevar.matmul_ff_q4_blocks_per_row, spevar.matmul_dim_row_worker_count);

    spevar.backlog = backlog_size;
    spevar.softmax_head_per_wavefront = workgroup_size / backlog_size;
    spevar.backlog_bits = ulog2(backlog_size);
    spevar.batch_enabled = (batch_size > 1) ? 1 : 0;
}

void llava_session::rewind(u32 n) {
    if (n < token_buffer.size()) {
        token_buffer.resize(n);
        current_tokens_in_gpu = n;
    }
}

u32 llava_session::get_token_count() const {
    return token_buffer.size();
}

#pragma clang diagnostic push
#pragma ide diagnostic ignored "ConstantFunctionResult"
ReturnCode llava_session::set_options(u32 new_options) {
    if (options == new_options) {
        return ReturnCode::ok;
    }

    options = new_options;

    delete command_buffer;
    command_buffer = nullptr;

    flush_layers_data_buffers();

    recreate_buffers();
    return ReturnCode::ok;
}
#pragma clang diagnostic pop

bool llava_session::is_tracing_enabled() const {
    return options != 0;
}

ReturnCode llava_session::save_frame(const string &path) {
    if (not is_tracing_enabled()) {
        return ReturnCode::not_tracing;
    }

    if (batch_size > 1) {
        return ReturnCode::batched_tick;
    }

    int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);

    if (fd < -1) {
        perror("open");
        return ReturnCode::nok;
    }

    u32 i = 0;

    for(llava_layer_session_data* l_data : layer_data) {
        l_data->dump_tracing_layers(fd, string("layer.") + to_string(i));
        ++i;
    }
    close(fd);

    return ReturnCode::ok;
}

ReturnCode llava_session::snapshot(const string &path) {
    size_t total_size = layer_data.size() * 2 * current_tokens_in_gpu * model->header.dim * 2 + 4 * current_tokens_in_gpu + 4;
    assert(current_tokens_in_gpu <= token_buffer.size());

    int fd = open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);

    if (fd < -1) {
        perror("open");
        return ReturnCode::nok;
    }

    if (ftruncate64(fd, (ssize_t)total_size) < 0) {
        perror("ftruncate64");
        if (close(fd) < 0) {
            perror("close");
        }
        return ReturnCode::nok;
    }

    void* mapping = mmap(nullptr, total_size, PROT_WRITE, MAP_SHARED, fd, 0);
    if (mapping == MAP_FAILED) {
        perror("mmap");
        if (close(fd) < 0) {
            perror("close");
        }
        return ReturnCode::nok;
    }

    *((u32*)mapping) = current_tokens_in_gpu;
    memcpy(((u32*)mapping) + 1, token_buffer.data(), current_tokens_in_gpu * 4);

    u8* cursor = ((u8*)mapping) + (4 * current_tokens_in_gpu + 4);
    for (llava_layer_session_data* l_data : layer_data) {
        l_data->dump_kv_cache(cursor, current_tokens_in_gpu);
        cursor += 2 * current_tokens_in_gpu * model->header.dim * 2;
    }

    if (msync(mapping, total_size, MS_SYNC) < 0) {
        perror("msync");
    }
    if (munmap(mapping, total_size) < 0) {
        perror("munmap");
    }
    if (close(fd) < 0) {
        perror("close");
    }

    return ReturnCode::ok;
}

ReturnCode llava_session::restore(const string &path) {
    assert(current_tokens_in_gpu <= token_buffer.size());

    int fd = open(path.c_str(), O_RDONLY);

    if (fd < -1) {
        perror("open");
        return ReturnCode::nok;
    }

    struct stat statbuf{};

    if (fstat(fd, &statbuf) < 0) {
        perror("fstat");
        if (close(fd) < 0) {
            perror("close");
        }
        return ReturnCode::nok;
    }

    if (not S_ISREG(statbuf.st_mode)) {
        cerr << "Expecetd regular file got something else" << endl;
        if (close(fd) < 0) {
            perror("close");
        }
        return ReturnCode::nok;
    }

    if (statbuf.st_size < 4) {
        cerr << "File too short" << endl;
        if (close(fd) < 0) {
            perror("close");
        }
        return ReturnCode::nok;
    }

    void* mapping = mmap(nullptr, statbuf.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (mapping == MAP_FAILED) {
        perror("mmap");
        if (close(fd) < 0) {
            perror("close");
        }
        return ReturnCode::nok;
    }

    u32 token_count = *((u32*)mapping);
    if (token_count > max_backlog_size) {
        cerr << "Too many tokens" << endl;
        if (munmap(mapping, statbuf.st_size) < 0) {
            perror("munmap");
        }
        if (close(fd) < 0) {
            perror("close");
        }
        return ReturnCode::nok;
    }

    u32 needed_backlog_size = min_backlog_size;
    while (needed_backlog_size < token_count) {
        needed_backlog_size <<= 1;
    }

    size_t expected_size = layer_data.size() * 2 * token_count * model->header.dim * 2 + 4 * token_count + 4;

    if (expected_size != statbuf.st_size) {
        cerr << "File size mismatch" << endl;
        if (munmap(mapping, statbuf.st_size) < 0) {
            perror("munmap");
        }
        if (close(fd) < 0) {
            perror("close");
        }
        return ReturnCode::nok;
    }

    rewind(0);
    if (not set_backlog_size(needed_backlog_size)) {
        cerr << "Cannot set backlog size" << endl;
        if (munmap(mapping, statbuf.st_size) < 0) {
            perror("munmap");
        }
        if (close(fd) < 0) {
            perror("close");
        }
        return ReturnCode::nok;
    }

    token_buffer.resize(token_count);
    memcpy(token_buffer.data(), ((u32*)mapping) + 1, 4 * token_count);
    current_tokens_in_gpu = token_count;

    u8 const* cursor = ((u8*)mapping) + (4 * current_tokens_in_gpu + 4);

    for (llava_layer_session_data* l_data : layer_data) {
        l_data->restore_kv_cache(cursor, token_count);
        cursor += 2 * token_count * model->header.dim * 2;
    }

    if (munmap(mapping, expected_size) < 0) {
        perror("munmap");
    }
    if (close(fd) < 0) {
        perror("close");
    }

    return ReturnCode::ok;
}

vector<u32> const &llava_session::get_token_buffer() const {
    return token_buffer;
}
