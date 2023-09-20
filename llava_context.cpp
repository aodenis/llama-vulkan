#include "llava_context.h"
#include "llava_pipeline.h"
#include "llava_device_memory.h"
#include "llava_layer.h"
#include "llava_command_buffer.h"
#include "llava_session.h"
#include "utils.h"
#include <iostream>
#include <csignal>
#include <vulkan/vulkan.hpp>
#include "ggml_file.h"
#include "server/server.h"
#include <cmath>
#include <set>
#include <sys/signalfd.h>

#ifdef RUNTIME_BUILD_ENABLED
#include <glslang/Public/ShaderLang.h>
#endif

#ifdef EMBEDDED_SPV
llava_context::llava_context() {
    auto const* packed_data = (packed_data_t const*)(raw_packed_shaders);
    auto const* packed_data_as_char = (char const*)(raw_packed_shaders);
    for (u32 i = 0; i < packed_data->count; i++) {
        auto& entry = packed_data->entries[i];
        assert((entry.data_offset % 4) == 0);
        string shader_name(packed_data_as_char + entry.name_offset);
        embedded_shaders.emplace(shader_name, pair((u32*)(raw_packed_shaders + entry.data_offset), entry.data_length + 0));
    }
}
#else
llava_context::llava_context() = default;
#endif

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
    named_pipelines.clear();
    layers.clear();

    if (device) {
        device.destroy(command_pool);
        device.destroy(descriptor_pool);
        device.destroy(pipeline_cache);
        command_pool = nullptr;
        descriptor_pool = nullptr;
        pipeline_cache = nullptr;
        device.destroy();
    }
    queue = nullptr;
    physical_device = nullptr;
    device = nullptr;
    if (vulkan_instance) {
        vulkan_instance.destroy();
    }

    vulkan_instance = nullptr;
    delete model;
    if (sigfd != -1) {
        close(sigfd);
        sigfd = -1;
    }
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
        if ((queueFamilyProperties.at(i).queueFlags & vk::QueueFlagBits::eCompute) and ((queueFamilyProperties.at(i).queueFlags & vk::QueueFlagBits::eGraphics) != vk::QueueFlagBits::eGraphics)) {
            return i;
        }
    }
    for (uint32_t i = 0; i < queueFamilyProperties.size(); i++) {
        if (queueFamilyProperties.at(i).queueFlags & vk::QueueFlagBits::eCompute) {
            return i;
        }
    }
    return ~0U;
}

bool streq(const char* a1, const char* a2) {
    return (strcmp(a1, a2) == 0);
}

int llava_context::run(int argc, char **argv) {
#ifndef RUNTIME_BUILD_ENABLED
    use_prebuilt_shaders = true;
#endif
    bool server_mode = false;
    bool controlled = false;
    bool debug_mode = false;
    bool only_print_header = false;
    string model_path;
    bool model_path_provided = false;
    string prompt = "The ten best monuments to see in Paris are";

    for (u32 i = 1; i < argc; ++i) {
        if (string(argv[i]) == "--use-prebuilt") {
            use_prebuilt_shaders = true;
        } else if (streq(argv[i], "--debug")) {
            debug_mode = true;
        } else if (streq(argv[i], "--print_header")) {
            only_print_header = true;
        } else if (streq(argv[i], "--model") or streq(argv[i], "-m")) {
            if (i + 1 >= argc) {
                cerr << "[!] Expected model path after " << argv[i] << endl;
                exit(1);
            }
            if (model_path_provided) {
                cerr << "[!] -m or --model duplicated in the command line" << endl;
            }
            model_path_provided = true;
            ++i;
            model_path = argv[i];
        } else if (streq(argv[i], "--help") or streq(argv[i], "-h")) {
            cout << (argc ? argv[0] : "./llama_vulkan") << " [-h] [-m model_name.bin] [prompt] [-r]" << endl;
            exit(0);
        } else if (streq(argv[i], "--verbose") or streq(argv[i], "-v")) {
            verbosity++;
        } else if (streq(argv[i], "--control") or streq(argv[i], "-c")) {
            controlled = true;
        } else if (streq(argv[i], "--server") or streq(argv[i], "-s")) {
            server_mode = true;
        } else if (streq(argv[i], "--sigdebug")) {
            signal_debug = true;
        } else {
            if (i + 1 != argc) {
                cerr << "[!] Unexpected argument " << argv[i] << endl;
                exit(1);
            } else {
                prompt = argv[i];
            }
        }
    }

    if (controlled) {
        cerr << "Control disabled for now" << endl;
        exit(1);
    }

    if (model_path.empty()) {
        const char *model_path_env = ::getenv("LLAVA_MODEL");
        if (model_path_env) {
            model_path = model_path_env;
        }
    }

    if (model_path.empty()) {
        cerr << "[!] No model path provided" << endl;
        exit(1);
    }

    model = new ggml_file(model_path.c_str());

    if (not model->is_open()) {
        return 1;
    }

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
    if (verbosity) {
        cout << "Selected device: " << physical_device.getProperties().deviceName << endl;
    }

    queueFamilyIndex = find_suitable_queue_index();
    if (!~queueFamilyIndex) {
        cerr << "[!] No compute queue family found on selected device" << endl;
        return 1;
    }
    if (verbosity >= 2) {
        cout << "Selected queue: " << queueFamilyIndex << endl;
    }

    mainMemoryTypeIndex = find_suitable_memory_type(physical_device);
    if (!~mainMemoryTypeIndex) {
        cerr << "[!] No suitable memory type found on selected device" << endl;
        return 1;
    }

    this->workgroup_size = physical_device.getProperties().limits.maxComputeWorkGroupInvocations;
    ulog2(this->workgroup_size); // Assert it is a pow2

    // create a Device
    float queuePriority = 0.0f;
    vk::DeviceQueueCreateInfo deviceQueueCreateInfo(vk::DeviceQueueCreateFlags(), queueFamilyIndex, 1, &queuePriority);
    vk::PhysicalDevice16BitStorageFeatures features16bit;
    features16bit.storageInputOutput16 = false;
    features16bit.uniformAndStorageBuffer16BitAccess = false;
    features16bit.storageBuffer16BitAccess = true;
    device = physical_device.createDevice(vk::DeviceCreateInfo(vk::DeviceCreateFlags(), deviceQueueCreateInfo, {}, {}, {}, &features16bit));

    // create a CommandPool to allocate a CommandBuffer from
    command_pool = device.createCommandPool({{}, queueFamilyIndex});

    // Descriptor pool
    vk::DescriptorPoolSize descriptorPoolSize(vk::DescriptorType::eStorageBuffer, 4096 * 16);
    descriptor_pool = device.createDescriptorPool({vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
                                                   descriptorPoolSize.descriptorCount, 1, &descriptorPoolSize});
    // Queue
    queue = device.getQueue(queueFamilyIndex, 0);

    // Pipeline cache
    pipeline_cache = device.createPipelineCache({{}, 0, nullptr});

    if (not setup_signal_handling()) {
        cerr << "[!] Cannot setup signal handling" << endl;
        return 1;
    }

    layers.reserve(model->header.n_layers);
    for (u32 i = 0; i < model->header.n_layers; ++i) {
        layers.emplace_back(this, i);
    }

    for (auto& layer : layers) {
        if (pop_signal()) {
            return 1;
        }
        layer.freeze_storage();
    }

    {
        list<u32> layers_to_load;
        mutex gpu_load_mutex;
        for (auto& layer : layers) {
            layers_to_load.push_back(layer.layer_id);
        }
        vector<thread> gpu_load_threads;
        for(u32 i = 0; i < 8; ++i) {
            gpu_load_threads.emplace_back([this, &layers_to_load, &gpu_load_mutex](){
                while (true) {
                    u32 to_load;
                    {
                        lock_guard guard(gpu_load_mutex);
                        if(layers_to_load.empty()) {
                            break;
                        }
                        to_load = layers_to_load.front();
                        layers_to_load.pop_front();
                    }
                    this->layers.at(to_load).load_to_gpu();
                }
            });
        }
        for (auto& t : gpu_load_threads) {
            t.join();
        }
    }

#ifdef RUNTIME_BUILD_ENABLED
    glslang::InitializeProcess();
#endif

    if (server_mode) {
        lsrv::llava_server server(this);
        server.serve_forever();
    } else {
        u32 eos_id = 2;
        llava_session session(this);

        if (not session.set_text(prompt)) {
            cerr << "[!] System prompt overflows backlog buffer !" << endl;
            exit(1);
        }

        cout << prompt << flush;
        u32 next_token = session.predict_next_token();
        while (true) {
            if (~next_token == 0) {
                cerr << "Unknown error" << endl;
                break;
            }
            if (pop_signal() or (next_token == eos_id)) {
                break;
            }

            if (not session.push_token(next_token)) {
                break; // Too many tokens
            }

            if (not session.start_next_token_prediction()) {
                cerr << "Cannot start processing, unknown error" << endl;
                break;
            }
            cout << model->tokens[next_token].text << flush;
            next_token = session.finish_next_token_prediction();
        }

        if (next_token != eos_id) {
            cout << model->tokens[next_token].text;
        }

        cout << endl;
    }

#ifdef RUNTIME_BUILD_ENABLED
    glslang::FinalizeProcess();
#endif

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

ggml_file const* llava_context::get_model() const {
    assert(model);
    return model;
}

using pipeline_signature = pair<string, specialization_variables_t>;

std::weak_ordering operator<=>(const pipeline_signature &lhs, const pipeline_signature &rhs) {
    auto f = lhs.first <=> rhs.first;
    if(f != std::weak_ordering::equivalent) {
        return f;
    }
    int r = memcmp(&lhs.second, &rhs.second, sizeof(rhs.second));
    if (r < 0) {
        return std::weak_ordering::less;
    }
    if (r == 0) {
        return std::weak_ordering::equivalent;
    }
    return std::weak_ordering::greater;
}

llava_pipeline *llava_context::get_pipeline(const string &shader_name, u32 argument_count, specialization_variables_t const& spevars) {
    lock_guard guard(pipeline_mutex);
    pair<string, specialization_variables_t> signature(shader_name, spevars);
    auto it = named_pipelines.find(signature);
    if (it != named_pipelines.end()) {
        assert (it->second.argument_count == argument_count);
        return &it->second;
    }

    it = named_pipelines.emplace(std::piecewise_construct, std::forward_as_tuple(signature), forward_as_tuple(this, shader_name, spevars, use_prebuilt_shaders, argument_count)).first;
    return &it->second;
}

vk::Queue &llava_context::get_queue() {
    assert(queue);
    return queue;
}

vector<llava_layer>& llava_context::get_layers() {
    return layers;
}

string llava_context::generate_spevar_define_string(specialization_variables_t const* spevars) {
    stringstream ss;
    ss << "#define HEAD_COUNT " << spevars->head_count << "\n";
    ss << "#define QUARTERROT " << spevars->quarterrot << "\n";
    ss << "#define BACKLOG " << spevars->backlog << "\n";
    ss << "#define MAX_WGS " << spevars->max_wgs << "\n";
    ss << "#define MAX_WGS_BITS " << spevars->max_wgs_bits << "\n";
    ss << "#define FF_SIZE " << spevars->ff_size << "\n";
    ss << "#define SOFTMAX_HEAD_PER_WAVEFRONT " << spevars->softmax_head_per_wavefront << "\n";
    ss << "#define BACKLOG_BITS " << spevars->backlog_bits << "\n";
    ss << "#define ROT_BITS " << spevars->rot_bits << "\n";
    ss << "#define ROT " << spevars->rot << "\n";
    ss << "#define MATMUL_DIM_ROW_PER_WAVEFRONT " << spevars->matmul_dim_row_per_wavefront << "\n";
    ss << "#define MATMUL_DIM_ROW_WORKER_COUNT " << spevars->matmul_dim_row_worker_count << "\n";
    ss << "#define MATMUL_DIM_ROW_WORKER_COUNT_LOG2 " << spevars->matmul_dim_row_worker_count_log2 << "\n";
    ss << "#define MATMUL_DIM_Q4_BLOCK_COUNT_PER_WORKER " << spevars->matmul_dim_q4_block_count_per_worker << "\n";
    ss << "#define MATMUL_DIM_Q4_BLOCKS_PER_ROW " << spevars->matmul_dim_q4_blocks_per_row << "\n";
    ss << "#define MATMUL_FF_ROW_PER_WAVEFRONT " << spevars->matmul_ff_row_per_wavefront << "\n";
    ss << "#define MATMUL_FF_ROW_WORKER_COUNT " << spevars->matmul_ff_row_worker_count << "\n";
    ss << "#define MATMUL_FF_ROW_WORKER_COUNT_LOG2 " << spevars->matmul_ff_row_worker_count_log2 << "\n";
    ss << "#define MATMUL_FF_Q4_BLOCK_COUNT_PER_WORKER " << spevars->matmul_ff_q4_block_count_per_worker << "\n";
    ss << "#define MATMUL_FF_Q4_BLOCKS_PER_ROW " << spevars->matmul_ff_q4_blocks_per_row << "\n";
    ss << "#define BATCH_ENABLED " << spevars->batch_enabled << "\n";
    return ss.str();
}

bool llava_context::setup_signal_handling() {
    if(signal_debug) {
        return true;
    }

    if (sigfd != -1) {
        return true;
    }
    sigset_t mask;

    sigemptyset(&mask);
    sigaddset(&mask, SIGINT);
    sigaddset(&mask, SIGTERM);
    sigaddset(&mask, SIGQUIT);
    sigaddset(&mask, SIGHUP);

    if (sigprocmask(SIG_BLOCK, &mask, nullptr) < 0) {
       perror("sigprocmask");
       return false;
    }

    sigfd = signalfd(-1, &mask, 0);
    if (sigfd == -1) {
       perror("signalfd");
       return false;
    }

    return true;
}

u32 llava_context::pop_signal(bool blocking) const {
    if (sigfd == -1) {
        if (not signal_debug) {
            cerr << "[?] pop_signal called but sigfd is unset" << endl;
            exit(1);
        } else return 0;
    }

    if (not blocking) {
        pollfd pfd{
            .fd = sigfd,
            .events = POLLIN,
            .revents = 0
        };
        int pollret = poll(&pfd, 1, 0);
        if (pollret == 0) {
            return 0;
        }
        if (pollret != 1) {
            perror("poll");
            return 0; // Either failed (wtf) or returned more than passed as arg (wtf)
        }
    }

    signalfd_siginfo fdsi{};
    if (read_noshort(sigfd, &fdsi, sizeof(fdsi)) < sizeof(fdsi)) {
        cerr << "[*] Short read on signal fd, wtf" << endl;
        exit(1);
    }
    return fdsi.ssi_signo;
}

int llava_context::get_signal_fd() const {
    return sigfd;
}

pair<u32 *, u32> llava_context::get_shader_spirv_by_name(const string &shader_name) {
    auto it = embedded_shaders.find(shader_name);
    if (it == embedded_shaders.end()) {
        return {nullptr, 0};
    }
    else {
        return it->second;
    }
}

bool llava_context::signal_debug_on() const {
    return signal_debug;
}
