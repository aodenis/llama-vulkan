#include "llava_context.h"
#include "llava_pipeline.h"
#include "llava_buffer.h"
#include "llava_device_memory.h"
#include "llava_layer.h"
#include "llava_command_buffer.h"
#include "utils.h"
#include <iostream>
#include <chrono>
#include <vulkan/vulkan.hpp>
#include "ggml_file.h"
#include <cmath>
#include <set>

#ifdef RUNTIME_BUILD_ENABLED
#include <glslang/Public/ShaderLang.h>
#endif

struct shared_state {
    u32 token_count;
    u32 dim;
    u32 pad1;
    u32 pad2;
};

vk::PhysicalDevice llava_context::find_suitable_physical_device() {
    auto physical_devices = vulkan_instance.enumeratePhysicalDevices();
    for (vk::PhysicalDevice const &pd : physical_devices) {
        if (pd.getProperties().deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
            if (find_suitable_memory_type(pd) != (~0U)) {
                return pd;
            } else if (verbosity > 0) {
                cout << "[*] Not using GPU " << pd.getProperties().deviceName << " as it does not meet memory requirements" << endl;
            }
        }
    }

    for (vk::PhysicalDevice const &pd : physical_devices) {
        if (pd.getProperties().deviceType == vk::PhysicalDeviceType::eIntegratedGpu) {
            if (find_suitable_memory_type(pd) != (~0U)) {
                return pd;
            } else if (verbosity > 0) {
                cout << "[*] Not using GPU " << pd.getProperties().deviceName << " as it does not meet memory requirements" << endl;
            }
        }
    }

    cerr << "[!] Cannot find suitable GPU" << endl;
    exit(1);
}

uint32_t llava_context::get_queue_family_index() const {
    return queueFamilyIndex;
}

vk::PhysicalDevice& llava_context::get_physical_device() {
    return physical_device;
}

llava_context::~llava_context() {
    delete command_buffer;
    command_buffer = nullptr;
    named_pipelines.clear();
    layers.clear();
    reset_main_buffers();

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

u32 llava_context::find_suitable_memory_type(vk::PhysicalDevice const& _physical_device) {
    auto memory_properties = _physical_device.getMemoryProperties();
    set<u32> accepted_memory_heaps;
    for(u32 i = 0; i < memory_properties.memoryHeapCount; i++) {
        if (memory_properties.memoryHeaps.at(i).size > model->mapping_size) {
            accepted_memory_heaps.insert(i);
        }
    }
    auto wanted_flags = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
    list<u32> device_not_local_types;
    for(u32 i = 0; i < memory_properties.memoryTypeCount; ++i) {
        auto& memType = memory_properties.memoryTypes.at(i);
        if (not accepted_memory_heaps.contains(memType.heapIndex)) {
            continue;
        }

        if ((memType.propertyFlags & wanted_flags) != wanted_flags) {
            continue;
        }

        if (memType.propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal) {
            return i;
        } else {
            device_not_local_types.push_back(i);
        }
    }

    if (not device_not_local_types.empty()) {
        return device_not_local_types.front();
    }

    return ~0U;
}

u32 llava_context::find_suitable_queue_index() {
    vector<vk::QueueFamilyProperties> queueFamilyProperties = physical_device.getQueueFamilyProperties();
    for (uint32_t i = 0; i < queueFamilyProperties.size(); i++) {
        if (queueFamilyProperties.at(i).queueFlags & vk::QueueFlagBits::eCompute) {
            return i;
        }
    }
    return ~0U;
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
        if (string(argv[i]) == "--use-prebuilt") {
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
            cout << (argc ? argv[0] : "./llama_vulkan") << " [-h] [-m model_name.bin] [prompt]" << endl;
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

    vk::ApplicationInfo applicationInfo("llava", 1, "llava0", 1, VK_API_VERSION_1_2);

    vector<const char *> enabled_layers;
    if (debug_mode) {
        enabled_layers.emplace_back("VK_LAYER_KHRONOS_validation");
    }

    // create an Instance
    vulkan_instance = vk::createInstance({{}, &applicationInfo, enabled_layers});
    physical_device = find_suitable_physical_device();

    queueFamilyIndex = find_suitable_queue_index();
    if (!~queueFamilyIndex) {
        cerr << "[!] No compute queue family found on selected device" << endl;
        return 1;
    }

    mainMemoryTypeIndex = find_suitable_memory_type(physical_device);
    if (!~mainMemoryTypeIndex) {
        cerr << "[!] No suitable memory type found on selected device" << endl;
        return 1;
    }

    this->workgroup_size = physical_device.getProperties().limits.maxComputeWorkGroupInvocations;
    ulog2(this->workgroup_size); // Assert it is a pow2
    if (verbosity) {
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
    u32 ff_size = model->ff_size;
    u32 n_heads =  model->header.n_heads;
    u32 rot = model->header.rot;
    auto& spevar = specialization_variables;

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

    vector<u32> dec_tokens;
    model->tokenize(dec_tokens, prompt, true);
    if (dec_tokens.size() > backlog_size) {
        cerr << "[!] System prompt overflows backlog buffer !" << endl;
        exit(-1);
    }

    for (u32 i = 0; i < model->header.n_layers; ++i) {
        layers.emplace_back(this, i);
    }

    set_batch_size(dec_tokens.size());

    for(auto& layer : layers) {
        layer.freeze_storage();
    }

    for(auto& layer : layers) {
        layer.load_from_disk();
    }

    u32 eos_id = 2;
    uint32_t next_token;
    auto start_time = std::chrono::high_resolution_clock::now();
    process_tokens(dec_tokens);
    next_token = get_last_predicted_token();
    while (tokens.size() < backlog_size) {
        uint32_t token = tokens.size() < dec_tokens.size() ? dec_tokens.at(tokens.size()) : next_token;
        if (token == eos_id) {
            break;
        }
        process_tokens({token});
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
    cout << tokens.size() << " tokens in " << (ns/1000000) << " milliseconds" << endl;
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

void llava_context::process_tokens(vector<u32> const& token_ids) {
    if (token_ids.size() != batch_size) {
        set_batch_size(token_ids.size());
    }
    if (command_buffer == nullptr) {
        command_buffer = new llava_command_buffer(this);
        command_buffer->record_execution();
    } else {
        command_buffer->reset_events();
    }

    ggml_data_descriptor const& descriptor = model->get_buffer_descriptor("tok_embeddings");
    assert((descriptor.size % model->tokens.size()) == 0);
    assert(tokens.size() + token_ids.size() <= backlog_size);

    for (u32 i = 0; i < token_ids.size(); i++) {
        u32 token_id = token_ids.at(i);
        assert(token_id < model->tokens.size());
        current_thought->write_f32(model->mapping + (descriptor.offset + token_id * (descriptor.size / model->tokens.size())), descriptor.ftype, i * model->header.dim, model->header.dim);
    }
    shared_state config {
        .token_count = static_cast<u32>(tokens.size()),
        .dim = model->header.dim,
    };
    config_buffer->write_full(&config, ggml_value_type::f32);
    command_buffer->run();
    for (u32 token_id : token_ids) {
        tokens.push_back(token_id);
    }
    queue.waitIdle();
}

u32 llava_context::get_last_predicted_token() const {
    auto* res = static_cast<float *>(output_probs->map(0, (batch_size - 1) * model->header.vocab_size * sizeof(float), model->header.vocab_size * sizeof(float)));
    u32 best = 0;
    for(uint32_t i = 1; i < model->header.vocab_size; ++i) {
        if (res[i] > res[best]) {
            best = i;
        }
    }
    output_probs->unmap();
    return best;
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


void llava_context::set_batch_size(u32 _batch_size) {
    if (batch_size == _batch_size) {
        return;
    }
    delete command_buffer;
    command_buffer = nullptr;
    batch_size = _batch_size;

    recreate_buffers();
}

void llava_context::recreate_buffers() {
    reset_main_buffers();
    if (batch_size == 0) {
        return;
    }

    u32 dim = model->header.dim;
    u32 ff_size = model->ff_size;
    u32 n_heads =  model->header.n_heads;
    u32 vocab_size = model->header.vocab_size;

    // Create main buffers
    main_buffer_memory = new llava_device_memory(this);
    current_thought = new llava_buffer(this, ggml_value_type::f32, dim, batch_size, main_buffer_memory);
    current_thought_sublayer = new llava_buffer(this, ggml_value_type::f32, dim, batch_size, main_buffer_memory);
    current_thought_middle_normd = new llava_buffer(this, ggml_value_type::f32, dim, batch_size, main_buffer_memory);
    properties_mask = new llava_buffer(this, ggml_value_type::f32, ff_size, batch_size, main_buffer_memory);
    properties_associated_values = new llava_buffer(this, ggml_value_type::f32, ff_size, batch_size, main_buffer_memory);
    current_Q = new llava_buffer(this, ggml_value_type::f32, dim, batch_size, main_buffer_memory);
    current_K = new llava_buffer(this, ggml_value_type::f32, dim, batch_size, main_buffer_memory);
    attn_result = new llava_buffer(this, ggml_value_type::f32, backlog_size, n_heads * batch_size, main_buffer_memory);
    config_buffer = new llava_buffer(this, ggml_value_type::f32, sizeof(shared_state) / sizeof(float), batch_size, main_buffer_memory);
    current_V = new llava_buffer(this, ggml_value_type::f32, dim, batch_size, main_buffer_memory);
    current_Vout = new llava_buffer(this, ggml_value_type::f32, dim, batch_size, main_buffer_memory);
    norm_w = new llava_buffer(this, model->get_buffer_descriptor("norm"), main_buffer_memory);
    output_w = new llava_buffer(this, model->get_buffer_descriptor("output"), main_buffer_memory);
    output_probs = new llava_buffer(this, ggml_value_type::f32, vocab_size, batch_size, main_buffer_memory);
    main_buffer_memory->freeze();
    norm_w->load_from_disk();
    output_w->load_from_disk();
}

void llava_context::reset_main_buffers() {
    delete current_thought;
    delete current_thought_sublayer;
    delete current_Q;
    delete current_thought_middle_normd;
    delete current_K;
    delete current_V;
    delete current_Vout;
    delete attn_result;
    delete config_buffer;
    delete norm_w;
    delete output_w;
    delete output_probs;
    delete properties_mask;
    delete properties_associated_values;
    delete main_buffer_memory;
    current_thought = nullptr;
    current_thought_sublayer = nullptr;
    current_thought_middle_normd = nullptr;
    current_Q = nullptr;
    current_K = nullptr;
    current_V = nullptr;
    current_Vout = nullptr;
    attn_result = nullptr;
    config_buffer = nullptr;
    norm_w = nullptr;
    output_w = nullptr;
    output_probs = nullptr;
    properties_mask = nullptr;
    properties_associated_values = nullptr;
    main_buffer_memory = nullptr;
}

vk::Queue &llava_context::get_queue() {
    assert(queue);
    return queue;
}

specialization_variables_t const &llava_context::get_spevar_struct() const {
    return specialization_variables;
}

list<llava_layer> const& llava_context::get_layers() const {
    return layers;
}
