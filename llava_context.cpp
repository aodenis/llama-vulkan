#include "llava_context.h"
#include "llava_pipeline.h"
#include "llava_buffer.h"
#include "llava_layer.h"
#include "utils.h"
#include <iostream>
#include <chrono>
#include <vulkan/vulkan.hpp>
#include "ggml_file.h"
#include <cmath>
#ifdef RUNTIME_BUILD_ENABLED
#include <glslang/Public/ShaderLang.h>
#endif

struct shared_state {
    u32 token_count;
    u32 dim;
    u32 pad1;
    u32 pad2;
};

vk::PhysicalDevice llava_context::get_physical_device() {
    for(vk::PhysicalDevice const &pd : vulkan_instance.enumeratePhysicalDevices()) {
        if (pd.getProperties().deviceType == vk::PhysicalDeviceType::eIntegratedGpu)
            return pd;
    }
    assert(false);
}

uint32_t llava_context::get_queue_family_index() const {
    return queueFamilyIndex;
}

llava_context::~llava_context() {
    named_pipelines.clear();
    command_buffer.clear();
    command_buffer_raw.clear();
    layers.clear();
    delete current_thought;
    delete current_thought_sublayer;
    delete current_Q;
    delete current_K;
    delete current_V;
    delete attn_result;
    delete config_buffer;
    delete norm_w;
    delete output_w;
    delete output_probs;
    delete properties_mask;
    delete properties_associated_values;
    current_thought = nullptr;
    current_thought_sublayer = nullptr;
    current_Q = nullptr;
    current_K = nullptr;
    current_V = nullptr;
    attn_result = nullptr;
    config_buffer = nullptr;
    norm_w = nullptr;
    output_w = nullptr;
    output_probs = nullptr;
    properties_mask = nullptr;
    properties_associated_values = nullptr;

    if (device) {
        device.destroy(command_pool);
        device.destroy(descriptor_pool);
        device.destroy(pipeline_cache);
        command_pool = nullptr;
        descriptor_pool = nullptr;
        pipeline_cache = nullptr;
        device.destroy();
    }

    device = nullptr;
    if (vulkan_instance) {
        vulkan_instance.destroy();
    }

    vulkan_instance = nullptr;
}

u32 ulog2(u32 n) {
    assert (n != 0);
    u32 i = 0;
    while (((n & 1) == 0)) {
        n >>= 1;
        i++;
    }
    assert(n == 1);
    return i;
}

int llava_context::run(int argc, char **argv) {
#ifndef RUNTIME_BUILD_ENABLED
    use_prebuilt_shaders = true;
#endif
    bool debug_mode = false;
    bool only_print_header = false;
    string model_path;
    bool model_path_provided = false;
    string prompt = "The ten best monuments to see in Paris are";

    for (u32 i = 1; i < argc; ++i) {
        if (string(argv[i]) == "--dell") {
            backupMemoryTypeIndex = 0;
            mainMemoryTypeIndex = 0;
        } else if (string(argv[i]) == "--use-prebuilt") {
            use_prebuilt_shaders = true;
        } else if (string(argv[i]) == "--debug") {
            debug_mode = true;
        } else if (string(argv[i]) == "--print_header") {
            only_print_header = true;
        } else if (string(argv[i]) == "--model" or string(argv[i]) == "-m") {
            if (i + 1 >= argc) {
                cerr << "[!] Expected model path after " << argv[i] << endl;
                exit(-1);
            }
            if (model_path_provided) {
                cerr << "[!] -m or --model duplicated in the command line" << endl;
            }
            model_path_provided = true;
            ++i;
            model_path = argv[i];
        } else if (string(argv[i]) == "--help" or string(argv[i]) == "-h") {
            cout << (argc ? argv[0] : "./llama_vulkan") << " [-h] [--dell] [-m model_name.bin] [prompt]" << endl;
            exit(0);
        } else if (string(argv[i]) == "--verbose" or string(argv[i]) == "-v") {
            verbosity++;
        } else {
            if (i + 1 != argc) {
                cerr << "[!] Unexpected argument " << argv[i] << endl;
                exit(-1);
            } else {
                prompt = argv[i];
            }
        }
    }

    if (model_path.empty()) {
        const char *model_path_env = ::getenv("LLAVA_MODEL");
        if (model_path_env) {
            model_path = model_path_env;
        }
    }

    if (model_path.empty()) {
        cerr << "[!] No model path provided" << endl;
        exit(-1);
    }

    model = std::make_shared<ggml_file>(model_path.c_str());

    if (only_print_header) {
        model->print_info();
        return 0;
    }

    vector<u32> dec_tokens;
    model->tokenize(dec_tokens, prompt, true);
    if (dec_tokens.size() >= backlog_size) {
        cerr << "[!] Prompt overflows backlog buffer !" << endl;
        exit(-1);
    }

    vk::ApplicationInfo applicationInfo("llava", 1, "llava0", 1, VK_API_VERSION_1_2);

    vector<const char *> enabled_layers;
    if (debug_mode) {
        enabled_layers.emplace_back("VK_LAYER_KHRONOS_validation");
    }

    // create an Instance
    vulkan_instance = vk::createInstance({{}, &applicationInfo, enabled_layers});
    physical_device = get_physical_device();

    {
        std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physical_device.getQueueFamilyProperties();
        // Compute queue family index
        for (uint32_t i = 0; i < queueFamilyProperties.size(); i++) {
            if (queueFamilyProperties.at(i).queueFlags & vk::QueueFlagBits::eCompute) {
                queueFamilyIndex = i;
                break;
            }
        }
    }
    assert(~queueFamilyIndex);

    this->workgroup_size = physical_device.getProperties().limits.maxComputeWorkGroupInvocations;
    ulog2(this->workgroup_size); // Assert it is a pow2
    if (verbosity > 0) {
        cout << "Selected device: " << physical_device.getProperties().deviceName << endl;
    }

    // create a Device
    float queuePriority = 0.0f;
    vk::DeviceQueueCreateInfo deviceQueueCreateInfo(vk::DeviceQueueCreateFlags(), queueFamilyIndex, 1, &queuePriority);
    device = physical_device.createDevice({vk::DeviceCreateFlags(), deviceQueueCreateInfo});

    // create a CommandPool to allocate a CommandBuffer from
    command_pool = device.createCommandPool({{}, queueFamilyIndex});

    // Descriptor pool
    vk::DescriptorPoolSize descriptorPoolSize(vk::DescriptorType::eStorageBuffer, 2048);
    descriptor_pool = device.createDescriptorPool({vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
                                                   2048, 1, &descriptorPoolSize});
    // Queue
    queue = device.getQueue(queueFamilyIndex, 0);

    // Pipeline cache
    pipeline_cache = device.createPipelineCache({{}, 0, nullptr});

#ifdef RUNTIME_BUILD_ENABLED
    glslang::InitializeProcess();
#endif

    // Compute specialization variables
    u32 dim = model->header.dim;
    assert(model->ff_size % 32 == 0);
    assert(dim % 32 == 0);
    assert(workgroup_size % backlog_size == 0);
    assert(model->header.rot > 2);

    specialization_variables.head_count = model->header.n_heads;
    specialization_variables.rot = model->header.rot;
    specialization_variables.quarterrot = model->header.rot / 4;
    specialization_variables.rot_bits = ulog2(model->header.rot);

    specialization_variables.backlog = backlog_size;
    specialization_variables.max_wgs = this->workgroup_size;
    specialization_variables.max_wgs_bits = ulog2(this->workgroup_size);
    specialization_variables.softmax_head_per_wavefront = workgroup_size / backlog_size;
    specialization_variables.ff_size = model->ff_size;
    specialization_variables.backlog_bits = ulog2(backlog_size);

    if (dim == 4096) {
        specialization_variables.matmul_dim_row_worker_count = 128;
    } else if (dim == 5120) {
        specialization_variables.matmul_dim_row_worker_count = 32;
    } else {
        assert(false);
    }

    assert((workgroup_size % specialization_variables.matmul_dim_row_worker_count) == 0);
    specialization_variables.matmul_dim_row_per_wavefront = workgroup_size / specialization_variables.matmul_dim_row_worker_count;
    specialization_variables.matmul_dim_row_worker_count_log2 = ulog2(specialization_variables.matmul_dim_row_worker_count);
    specialization_variables.matmul_dim_q4_blocks_per_row = model->header.dim / 32;
    specialization_variables.matmul_dim_q4_block_count_per_worker = updiv(specialization_variables.matmul_dim_q4_blocks_per_row, specialization_variables.matmul_dim_row_worker_count);

    if ((model->ff_size == 11008) or (model->ff_size == 13824)) {
        specialization_variables.matmul_ff_row_worker_count = 128;
    } else {
        assert(false);
    }
    assert((workgroup_size % specialization_variables.matmul_ff_row_worker_count) == 0);
    specialization_variables.matmul_ff_row_per_wavefront = workgroup_size / specialization_variables.matmul_ff_row_worker_count;
    specialization_variables.matmul_ff_row_worker_count_log2 = ulog2(specialization_variables.matmul_ff_row_worker_count);
    specialization_variables.matmul_ff_q4_blocks_per_row = model->ff_size / 32;
    specialization_variables.matmul_ff_q4_block_count_per_worker = updiv(specialization_variables.matmul_ff_q4_blocks_per_row, specialization_variables.matmul_dim_row_worker_count);

    current_thought = new llava_buffer(this, "current_thought", ggml_value_type::f32, model->header.dim);
    current_thought_sublayer = new llava_buffer(this, "current_thought_sublayer", ggml_value_type::f32, model->header.dim);
    properties_mask = new llava_buffer(this, "properties_mask", ggml_value_type::f32, model->ff_size, 1, 1024);
    properties_associated_values = new llava_buffer(this, "properties_associated_values", ggml_value_type::f32, model->ff_size, 1, 1024);
    current_Q = new llava_buffer(this, "current_Q", ggml_value_type::f32, model->header.dim);
    current_K = new llava_buffer(this, "current_K", ggml_value_type::f32, model->header.dim);
    attn_result = new llava_buffer(this, "attn_result", ggml_value_type::f32, backlog_size, model->header.n_heads);
    config_buffer = new llava_buffer(this, "config_buffer", ggml_value_type::f32, sizeof(shared_state) / sizeof(float), 1, 1);
    current_V = new llava_buffer(this, "current_V", ggml_value_type::f32, model->header.dim);
    norm_w = new llava_buffer(this, model->get_buffer_descriptor("norm"));
    output_w = new llava_buffer(this, model->get_buffer_descriptor("output"));
    output_probs = new llava_buffer(this, "output_probs", ggml_value_type::f32, model->header.vocab_size);

    for (u32 i = 0; i < model->header.n_layers; ++i) {
        layers.emplace_back(this, i);
    }
    u32 eos_id = 13;
    record_execution(nullptr);
    uint32_t next_token = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    while (tokens.size() < backlog_size) {
        reset_command_buffer_events();
        uint32_t token = tokens.size() < dec_tokens.size() ? dec_tokens.at(tokens.size()) : next_token;
        if (token == eos_id) {
            break;
        }
        process_token(token);
        queue.waitIdle();
        next_token = get_last_predicted_token();
        vector<char> token_value = model->tokens[token].text;
        vector<char> next_token_value = model->tokens[next_token].text;
        token_value.push_back(0);
        next_token_value.push_back(0);
        // cout << token_value.data() << " -> " << next_token_value.data() << " (" << token << " -> " << next_token << ")\n";
        cout << token_value.data();
        cout << flush;
    }
    if (next_token != eos_id) {
        vector<char> last_token_value = model->tokens[next_token].text;
        last_token_value.push_back(0);
        cout << last_token_value.data();
    }
    cout << endl;
    auto end_time = std::chrono::high_resolution_clock::now();
    u64 ns = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();
    cout << tokens.size() << " tokens in " << ns << " nanoseconds" << endl;
    cout << (ns/(1000000*tokens.size())) << " milliseconds per token" << endl;
    return 0;
}

vk::Device& llava_context::get_device() {
    assert(device);
    return device;
}

vk::CommandPool& llava_context::get_command_pool() {
    assert(command_pool);
    return command_pool;
}

vk::DescriptorPool& llava_context::get_descriptor_pool() {
    assert(descriptor_pool);
    return descriptor_pool;
}

vk::PipelineCache& llava_context::get_pipeline_cache() {
    assert(pipeline_cache);
    return pipeline_cache;
}

shared_ptr<ggml_file> llava_context::get_model() {
    assert(model != nullptr);
    return model;
}

llava_pipeline *llava_context::get_pipeline(const string &shader_name, u32 argcount) {
    auto it = named_pipelines.find(shader_name);
    if (it != named_pipelines.end()) {
        assert (it->second.argcount == argcount);
        return &it->second;
    }

    it = named_pipelines.emplace(std::piecewise_construct, forward_as_tuple(shader_name), forward_as_tuple(this, shader_name, use_prebuilt_shaders, argcount)).first;
    return &it->second;
}

void llava_context::process_token(u32 token_id) {
    assert(token_id < model->tokens.size());

    ggml_data_descriptor const& descriptor = model->get_buffer_descriptor("tok_embeddings");
    assert((descriptor.size % model->tokens.size()) == 0);
    current_thought->write_full(model->mapping + (descriptor.offset + token_id * (descriptor.size / model->tokens.size())), descriptor.ftype);

    shared_state config {
        .token_count = static_cast<u32>(tokens.size()),
        .dim = model->header.dim,
    };
    config_buffer->write_full(&config, ggml_value_type::f32);
    tokens.push_back(token_id);
    vk::SubmitInfo submitInfo({}, {}, command_buffer_raw, {}, nullptr);
    queue.submit({submitInfo});
    queue.waitIdle();
}

vk::Event llava_context::record_execution(vk::Event startEvent) {
    command_buffer_raw.clear();
    command_buffer.clear();

    for (auto& layer : layers) {
        startEvent = layer.execute(this, {startEvent});
    }

    startEvent = normalize_logit(current_thought_sublayer, current_thought, norm_w, {startEvent});
    startEvent = matmul(output_probs, output_w, current_thought_sublayer, {startEvent});

    command_buffer_raw.reserve(command_buffer.size());
    for (auto& command : command_buffer) {
        command_buffer_raw.push_back(command.commandBuffer);
    }
    return startEvent;
}

u32 llava_context::get_last_predicted_token() const {
    auto* res = static_cast<float *>(device.mapMemory(output_probs->deviceMemory, 0, output_probs->size));
    u32 best = 0;
    for(uint32_t i = 1; i < model->header.vocab_size; ++i) {
        if (res[i] > res[best]) {
            best = i;
        }
    }
    device.unmapMemory(output_probs->deviceMemory);
    return best;
}

vk::Event llava_context::normalize_logit(llava_buffer* outbuf, llava_buffer* inbuf, llava_buffer* weights, initializer_list<vk::Event> events) {
    return record_command("normalize", {outbuf, inbuf, weights}, events, 1);
}

vk::Event llava_context::row_wise_multiply(llava_buffer* buf, llava_buffer* weights, initializer_list<vk::Event> events) {
    assert(buf->shape == weights->shape);
    assert(buf->shape.second == 1);
    return record_command("multiply", {buf, weights}, events, updiv(weights->shape.first, workgroup_size));
}

vk::Event llava_context::matmul(llava_buffer* outbuf, llava_buffer* matrix, llava_buffer* inbuf, initializer_list<vk::Event> events) {
    assert(inbuf->shape.first == matrix->shape.second);
    assert(outbuf->shape.first == matrix->shape.first);
    assert(inbuf->shape.second == 1);
    assert(outbuf->shape.second == 1);

    if (inbuf->shape.first == model->header.dim) {
        assert(outbuf->shape.first % specialization_variables.matmul_dim_row_per_wavefront == 0);
        return record_command("matmul_dim", {outbuf, matrix, inbuf}, events, outbuf->shape.first / specialization_variables.matmul_dim_row_per_wavefront);
    } else if (inbuf->shape.first == model->ff_size) {
        assert(outbuf->shape.first % specialization_variables.matmul_ff_row_per_wavefront == 0);
        return record_command("matmul_ff", {outbuf, matrix, inbuf}, events, outbuf->shape.first / specialization_variables.matmul_ff_row_per_wavefront);
    } else {
        cout << "MATMUL " << matrix->shape.first << "," << matrix->shape.second << " " << ftype_name(outbuf->type) << " " << ftype_name(matrix->type) << " " << ftype_name(inbuf->type) << endl;
        return nullptr;
    }
}

vk::Event llava_context::kv_copy(llava_buffer* out_cache, llava_buffer* input_line, initializer_list<vk::Event> events) {
    assert(out_cache->shape.first == backlog_size);
    assert(out_cache->shape.second == model->header.dim);
    assert(input_line->shape.first == model->header.dim);
    assert(input_line->shape.second == 1);
    return record_command("copy_to_cache", {config_buffer, out_cache, input_line}, events, model->header.dim);
}

vk::Event llava_context::multi_head_attention(llava_buffer* out_buffer, llava_buffer* cache_buffer, llava_buffer* query, initializer_list<vk::Event> events) {
    assert(cache_buffer->shape.first == backlog_size);
    assert(cache_buffer->shape.second == model->header.dim);

    assert(query->shape.first == model->header.dim);
    assert(query->shape.second == 1);

    assert(out_buffer->shape.first == backlog_size);
    assert(out_buffer->shape.second == model->header.n_heads);

    assert((model->header.n_heads * backlog_size) % workgroup_size == 0);
    return record_command("mhsa", {out_buffer, cache_buffer, query}, events, model->header.n_heads * backlog_size / workgroup_size);
}

vk::Event llava_context::inplace_softmax(llava_buffer* inout_buffer, initializer_list<vk::Event> events) {
    assert(inout_buffer->shape.first == backlog_size);
    assert(inout_buffer->shape.second == model->header.n_heads);

    assert(model->header.n_heads % specialization_variables.softmax_head_per_wavefront == 0);
    return record_command("softmax", {config_buffer, inout_buffer}, events, model->header.n_heads / specialization_variables.softmax_head_per_wavefront);
}

vk::Event llava_context::add(llava_buffer* buf, llava_buffer* delta_buf, initializer_list<vk::Event> events) {
    return record_command("add_logits", {buf, delta_buf}, events, updiv(model->header.dim, workgroup_size));
}

vk::Event llava_context::silu(llava_buffer* buf, initializer_list<vk::Event> events) {
    return record_command("silu", {buf}, events, updiv(buf->shape.first, workgroup_size));
}

vk::Event llava_context::rope(llava_buffer* buf, initializer_list<vk::Event> events) {
    assert((model->header.rot % 2) == 0);
    return record_command("rope", {config_buffer, buf}, events, updiv(buf->shape.first / 2, workgroup_size));
}

vk::Event llava_context::perform_kqv_matching(llava_buffer* v_out, llava_buffer* v_cache, llava_buffer* softmax_out, initializer_list<vk::Event> events) {
    assert(v_out and v_cache and softmax_out);

    assert(v_out->shape.first == model->header.dim);
    assert(v_out->shape.second == 1);

    assert(v_cache->shape.first == backlog_size);
    assert(v_cache->shape.second == model->header.dim);

    assert(softmax_out->shape.first == backlog_size);
    assert(softmax_out->shape.second == model->header.n_heads);

    return record_command("kqv_matching", {v_out, v_cache, softmax_out}, events, model->header.dim / 8);
}

vk::Event llava_context::record_command(llava_pipeline *pipeline,
                                      const initializer_list<llava_buffer *> &buffers,
                                      const initializer_list<vk::Event> &events,
                                      uint32_t countX,
                                      uint32_t countY,
                                      uint32_t countZ) {
    return command_buffer.emplace_back(pipeline, buffers, events, countX, countY, countZ).completionEvent;
}

void llava_context::reset_command_buffer_events() {
    for (auto& x : command_buffer) {
        device.resetEvent(x.completionEvent);
    }
}

vk::Event llava_context::record_command(const string &pipeline_name, const initializer_list<llava_buffer *> &buffers, const initializer_list<vk::Event> &events, uint32_t countX, uint32_t countY,
                                      uint32_t countZ) {
    u32 buffer_count = 0;
    for(llava_buffer * buffer : buffers) {
        buffer_count += buffer->buffers.size();
    }
    auto* pipeline = get_pipeline(pipeline_name, buffer_count);
    return record_command(pipeline, buffers, events, countX, countY, countZ);
}

string llava_context::generate_spevar_define_string() const {
    stringstream ss;
    ss << "#define HEAD_COUNT " << specialization_variables.head_count << "\n";
    ss << "#define QUARTERROT " << specialization_variables.quarterrot << "\n";
    ss << "#define BACKLOG " << specialization_variables.backlog << "\n";
    ss << "#define MAX_WGS " << specialization_variables.max_wgs << "\n";
    ss << "#define MAX_WGS_BITS " << specialization_variables.max_wgs_bits << "\n";
    ss << "#define FF_SIZE " << specialization_variables.ff_size << "\n";
    ss << "#define SOFTMAX_HEAD_PER_WAVEFRONT " << specialization_variables.softmax_head_per_wavefront << "\n";
    ss << "#define BACKLOG_BITS " << specialization_variables.backlog_bits << "\n";
    ss << "#define ROT_BITS " << specialization_variables.rot_bits << "\n";
    ss << "#define ROT " << specialization_variables.rot << "\n";
    ss << "#define MATMUL_DIM_ROW_PER_WAVEFRONT " << specialization_variables.matmul_dim_row_per_wavefront << "\n";
    ss << "#define MATMUL_DIM_ROW_WORKER_COUNT " << specialization_variables.matmul_dim_row_worker_count << "\n";
    ss << "#define MATMUL_DIM_ROW_WORKER_COUNT_LOG2 " << specialization_variables.matmul_dim_row_worker_count_log2 << "\n";
    ss << "#define MATMUL_DIM_Q4_BLOCK_COUNT_PER_WORKER " << specialization_variables.matmul_dim_q4_block_count_per_worker << "\n";
    ss << "#define MATMUL_DIM_Q4_BLOCKS_PER_ROW " << specialization_variables.matmul_dim_q4_blocks_per_row << "\n";
    ss << "#define MATMUL_FF_ROW_PER_WAVEFRONT " << specialization_variables.matmul_ff_row_per_wavefront << "\n";
    ss << "#define MATMUL_FF_ROW_WORKER_COUNT " << specialization_variables.matmul_ff_row_worker_count << "\n";
    ss << "#define MATMUL_FF_ROW_WORKER_COUNT_LOG2 " << specialization_variables.matmul_ff_row_worker_count_log2 << "\n";
    ss << "#define MATMUL_FF_Q4_BLOCK_COUNT_PER_WORKER " << specialization_variables.matmul_ff_q4_block_count_per_worker << "\n";
    ss << "#define MATMUL_FF_Q4_BLOCKS_PER_ROW " << specialization_variables.matmul_ff_q4_blocks_per_row << "\n";
    return ss.str();
}
