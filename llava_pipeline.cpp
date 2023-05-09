#include "llava_pipeline.h"
#include "llava_context.h"
#include <utility>
#include <vulkan/vulkan.hpp>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#ifdef RUNTIME_BUILD_ENABLED
#include <utility>
#include <glslang/Public/ShaderLang.h>
#include <glslang/Public/ResourceLimits.h>
#include <glslang/SPIRV/GlslangToSpv.h>
#endif


vector<char> slurp_file(const string& path) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        cerr << "Cannot read " << path << endl;
        exit(1);
    }
    vector<char> out;
    while(true) {
        uint32_t cur = out.size();
        out.resize(out.size() + 1024);
        ssize_t r = read(fd, out.data() + cur, 1024);
        assert(r >= 0);
        out.resize(cur + r);
        if (r == 0) {
            close(fd);
            return out;
        }
    }
}
#ifdef RUNTIME_BUILD_ENABLED

class CustomIncluder : public glslang::TShader::Includer {
public:
    explicit CustomIncluder(string constants_values) : _const_val(std::move(constants_values)) {

    }

    IncludeResult* includeLocal(const char* headerName, const char* includerName, size_t inclusionDepth) override {
        if (headerName == nullptr or includerName == nullptr) {
            return nullptr;
        }

        string s_headerName(headerName);
        if (s_headerName == "constants.glsl") {
            char* raw_data = new char[_const_val.size()];
            ::memcpy(raw_data, _const_val.data(), _const_val.size());
            return new IncludeResult(s_headerName, raw_data, _const_val.size(), nullptr);
        } else {
            // TODO prevent trivial LFI
            vector<char> header_content = slurp_file(string("shaders/") + headerName);
            char* raw_data = new char[header_content.size()];
            ::memcpy(raw_data, header_content.data(), header_content.size());
            return new IncludeResult(s_headerName, raw_data, header_content.size(), nullptr);
        }
    }

    void releaseInclude(IncludeResult* result) override {
        delete[] result->headerData;
    }

private:
    const string _const_val;
};

static TBuiltInResource default_resources(llava_context* ctx)
{
    vk::PhysicalDevice& physical_device = ctx->get_physical_device();
    vk::PhysicalDeviceProperties props = physical_device.getProperties();
    auto& limits = props.limits;

    return {
            .maxLights                                 = 32,
            .maxClipPlanes                             = 6,
            .maxTextureUnits                           = 32,
            .maxTextureCoords                          = 32,
            .maxVertexAttribs                          = 64,
            .maxVertexUniformComponents                = 4096,
            .maxVaryingFloats                          = 64,
            .maxVertexTextureImageUnits                = 32,
            .maxCombinedTextureImageUnits              = 80,
            .maxTextureImageUnits                      = 32,
            .maxFragmentUniformComponents              = 4096,
            .maxDrawBuffers                            = 32,
            .maxVertexUniformVectors                   = 128,
            .maxVaryingVectors                         = 8,
            .maxFragmentUniformVectors                 = 16,
            .maxVertexOutputVectors                    = 16,
            .maxFragmentInputVectors                   = 15,
            .minProgramTexelOffset                     = -8,
            .maxProgramTexelOffset                     = 7,
            .maxClipDistances                          = static_cast<int>(limits.maxClipDistances),
            .maxComputeWorkGroupCountX                 = static_cast<int>(limits.maxComputeWorkGroupCount.at(0)),
            .maxComputeWorkGroupCountY                 = static_cast<int>(limits.maxComputeWorkGroupCount.at(1)),
            .maxComputeWorkGroupCountZ                 = static_cast<int>(limits.maxComputeWorkGroupCount.at(2)),
            .maxComputeWorkGroupSizeX                  = static_cast<int>(limits.maxComputeWorkGroupSize.at(0)),
            .maxComputeWorkGroupSizeY                  = static_cast<int>(limits.maxComputeWorkGroupSize.at(1)),
            .maxComputeWorkGroupSizeZ                  = static_cast<int>(limits.maxComputeWorkGroupSize.at(2)),
            .maxComputeUniformComponents               = 1024,
            .maxComputeTextureImageUnits               = 16,
            .maxComputeImageUniforms                   = 8,
            .maxComputeAtomicCounters                  = 8,
            .maxComputeAtomicCounterBuffers            = 1,
            .maxVaryingComponents                      = 60,
            .maxVertexOutputComponents                 = static_cast<int>(limits.maxVertexOutputComponents), // 64,
            .maxGeometryInputComponents                = static_cast<int>(limits.maxGeometryInputComponents), // 64,
            .maxGeometryOutputComponents               = static_cast<int>(limits.maxGeometryOutputComponents), // 128,
            .maxFragmentInputComponents                = static_cast<int>(limits.maxFragmentInputComponents), // 128,
            .maxImageUnits                             = 8,
            .maxCombinedImageUnitsAndFragmentOutputs   = 8,
            .maxCombinedShaderOutputResources          = 8,
            .maxImageSamples                           = 0,
            .maxVertexImageUniforms                    = 0,
            .maxTessControlImageUniforms               = 0,
            .maxTessEvaluationImageUniforms            = 0,
            .maxGeometryImageUniforms                  = 0,
            .maxFragmentImageUniforms                  = 8,
            .maxCombinedImageUniforms                  = 8,
            .maxGeometryTextureImageUnits              = 16,
            .maxGeometryOutputVertices                 = static_cast<int>(limits.maxGeometryOutputVertices), // 256,
            .maxGeometryTotalOutputComponents          = static_cast<int>(limits.maxGeometryTotalOutputComponents), // 1024,
            .maxGeometryUniformComponents              = 1024,
            .maxGeometryVaryingComponents              = 64,
            .maxTessControlInputComponents             = 128,
            .maxTessControlOutputComponents            = 128,
            .maxTessControlTextureImageUnits           = 16,
            .maxTessControlUniformComponents           = 1024,
            .maxTessControlTotalOutputComponents       = 4096,
            .maxTessEvaluationInputComponents          = 128,
            .maxTessEvaluationOutputComponents         = 128,
            .maxTessEvaluationTextureImageUnits        = 16,
            .maxTessEvaluationUniformComponents        = 1024,
            .maxTessPatchComponents                    = 120,
            .maxPatchVertices                          = 32,
            .maxTessGenLevel                           = 64,
            .maxViewports                              = static_cast<int>(limits.maxViewports), // 16,
            .maxVertexAtomicCounters                   = 0,
            .maxTessControlAtomicCounters              = 0,
            .maxTessEvaluationAtomicCounters           = 0,
            .maxGeometryAtomicCounters                 = 0,
            .maxFragmentAtomicCounters                 = 8,
            .maxCombinedAtomicCounters                 = 8,
            .maxAtomicCounterBindings                  = 1,
            .maxVertexAtomicCounterBuffers             = 0,
            .maxTessControlAtomicCounterBuffers        = 0,
            .maxTessEvaluationAtomicCounterBuffers     = 0,
            .maxGeometryAtomicCounterBuffers           = 0,
            .maxFragmentAtomicCounterBuffers           = 1,
            .maxCombinedAtomicCounterBuffers           = 1,
            .maxAtomicCounterBufferSize                = 16384,
            .maxTransformFeedbackBuffers               = 4,
            .maxTransformFeedbackInterleavedComponents = 64,
            .maxCullDistances                          = static_cast<int>(limits.maxCullDistances), // 8,
            .maxCombinedClipAndCullDistances           = static_cast<int>(limits.maxCombinedClipAndCullDistances), // 8,
            .maxSamples                                = 4,
            .maxMeshOutputVerticesNV                   = 256,
            .maxMeshOutputPrimitivesNV                 = 512,
            .maxMeshWorkGroupSizeX_NV                  = 32,
            .maxMeshWorkGroupSizeY_NV                  = 1,
            .maxMeshWorkGroupSizeZ_NV                  = 1,
            .maxTaskWorkGroupSizeX_NV                  = 32,
            .maxTaskWorkGroupSizeY_NV                  = 1,
            .maxTaskWorkGroupSizeZ_NV                  = 1,
            .maxMeshViewCountNV                        = 4,
            .limits = {
                    .nonInductiveForLoops                 = false,
                    .whileLoops                           = false,
                    .doWhileLoops                         = false,
                    .generalUniformIndexing               = true,
                    .generalAttributeMatrixVectorIndexing = true,
                    .generalVaryingIndexing               = true,
                    .generalSamplerIndexing               = true,
                    .generalVariableIndexing              = true,
                    .generalConstantMatrixVectorIndexing  = true,
            }};
}
#endif

llava_pipeline::llava_pipeline(llava_context* ctx,
                               string _shader_name,
                               bool use_prebuilt_shaders,
                               uint32_t argument_count) : argcount(argument_count),
                                                          context(ctx),
                                                          shader_name(std::move(_shader_name)) {
    vector<vk::DescriptorSetLayoutBinding> bindings;
    bindings.reserve(argument_count);
    while(bindings.size() < argument_count) {
        bindings.emplace_back(bindings.size(), vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute);
    }

    descriptorSetLayout = ctx->get_device().createDescriptorSetLayout({{}, argument_count, bindings.data()});
    pipelineLayout = ctx->get_device().createPipelineLayout({{}, 1, &descriptorSetLayout});

    if (use_prebuilt_shaders) {
        vector<char> spirv = slurp_file(string("prebuilt_shaders/") + shader_name + ".spv");
        shaderModule = ctx->get_device().createShaderModule({{}, (u32)(spirv.size()), (uint32_t *)(spirv.data())});

        vector<vk::SpecializationMapEntry> entries;
        for (u32 i = 0; i * 4 < sizeof (specialization_variables_t); ++i) {
            entries.emplace_back(i, i * 4, 4);
        }
        vk::SpecializationInfo speInfo(entries.size(), entries.data(), sizeof(specialization_variables_t), &(context->specialization_variables));
        pipeline = ctx->get_device().createComputePipeline(ctx->get_pipeline_cache(),
                                                           {{}, {{}, vk::ShaderStageFlagBits::eCompute, shaderModule, "main", &speInfo}, pipelineLayout}).value;
    } else {
#ifdef RUNTIME_BUILD_ENABLED
        vector<string> little_source_paths;
        little_source_paths.emplace_back(shader_name + ".glsl");

        vector<vector<char>> shader_sources;
        shader_sources.reserve(little_source_paths.size());
        for(auto& x : little_source_paths) {
            shader_sources.emplace_back(slurp_file(string("shaders/") + x));
        }

        vector<const char*> c_names;
        vector<const char*> c_source;
        vector<int> c_source_sizes;
        for(u32 i = 0; i < little_source_paths.size(); i++) {
            c_names.push_back(little_source_paths.at(i).c_str());
            c_source.push_back(shader_sources.at(i).data());
            c_source_sizes.push_back(static_cast<int>(shader_sources.at(i).size()));
        }


        glslang::TShader shader(EShLangCompute);
        shader.setStringsWithLengthsAndNames(c_source.data(), c_source_sizes.data(), c_names.data(), static_cast<int>(c_names.size()));
        shader.setEntryPoint("main");
        shader.setSourceEntryPoint("main");
        shader.setAutoMapBindings(true);
        shader.setAutoMapLocations(true);
        shader.setEnvInput(glslang::EShSourceGlsl, EShLangCompute, glslang::EShClientVulkan, 100);
        shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_2);
        shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_4);

        CustomIncluder includer(ctx->generate_spevar_define_string());
        TBuiltInResource resources = default_resources(ctx);

        auto messages = (EShMessages)(EShMsgSpvRules | EShMsgVulkanRules);
        string preprocessedSource;
        if (!shader.preprocess(&resources, 450, ENoProfile, false, false, messages, &preprocessedSource, includer)) {
            cerr << "Shader preprocessing failed" << endl;
            cerr << shader.getInfoLog() << endl;
            cerr << shader.getInfoDebugLog() << endl;
            exit(-1);
        }

        const char* preprocessedShaderStrings[1] = { preprocessedSource.c_str() };

        shader.setStrings(preprocessedShaderStrings, 1);
        if (!shader.parse(&resources, 450, false, messages)) {
            cerr << "Shader parsing failed" << endl;
            cerr << shader.getInfoLog() << endl;
            cerr << shader.getInfoDebugLog() << endl;
            exit(-1);
        }

        glslang::TProgram program;
        program.addShader(&shader);
        if (!program.link(messages)) {
            cerr << "Shader linking failed" << endl;
        }
        vector<uint32_t> spirv;
        glslang::GlslangToSpv(*program.getIntermediate(EShLanguage::EShLangCompute), spirv);
        shaderModule = ctx->get_device().createShaderModule({{}, spirv.size() * sizeof(uint32_t), spirv.data()});

        pipeline = ctx->get_device().createComputePipeline(ctx->get_pipeline_cache(),
                                                           {{}, {{}, vk::ShaderStageFlagBits::eCompute, shaderModule, "main", nullptr}, pipelineLayout}).value;
#else
        cerr << "[!] Runtime shader compilation is not enabled!" << endl;
        exit(-1);
#endif
    }
}

llava_pipeline::~llava_pipeline() {
    context->get_device().destroy(pipeline);
    context->get_device().destroy(pipelineLayout);
    context->get_device().destroy(descriptorSetLayout);
    context->get_device().destroy(shaderModule);
}
