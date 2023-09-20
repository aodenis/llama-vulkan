#ifndef VULKAN_LLAMA_GGML_FILE_H
#define VULKAN_LLAMA_GGML_FILE_H

#include <cstdint>
#include <vector>
#include <queue>
#include <string>
#include <cassert>
#include <map>
#include "types.h"

struct ggml_header {
    u32 magic;
    u32 file_version;
    u32 vocab_size;
    u32 dim;
    u32 multiple_of;
    u32 n_heads;
    u32 n_layers;
    u32 rot;
    u32 ftype;
} __attribute__((__packed__));

struct ggml_token {
    string text;
    float score;
};

class ggml_data_descriptor {
public:
    ggml_data_descriptor(std::string name, ggml_value_type ftype, u32 model_version, size_t offset, int32_t shape1, int32_t shape2);
    const std::string name;
    const ggml_value_type ftype;
    const u32 model_version;
    const size_t offset;
    const u32 shape1;
    const u32 shape2;
    const size_t size;

    [[nodiscard]] size_t size_in_file() const;
};

class ggml_file {
    friend class llava_context;
    friend class llava_buffer;
    friend class llava_layer;
    friend class llava_session;
    friend class llava_command_buffer;
    friend class llava_layer_session_data;
public:
    explicit ggml_file(const char* filepath);
    ~ggml_file();
    void print_info() const;
    void tokenize(std::vector<uint32_t> &output, const std::string &text, bool bos) const;
    string tokens_to_text(u32 const* tokens, u32 count) const;
    [[nodiscard]] ggml_data_descriptor const& get_buffer_descriptor(const string& s) const;
    [[nodiscard]] std::vector<ggml_token> const& get_tokens() const;

private:
    uint8_t* mapping = nullptr;
    std::size_t mapping_size = 0;

private:
    ggml_header header;
    u32 ff_size;
    std::vector<ggml_token> tokens;
    std::vector<ggml_data_descriptor> tables;
    map<string, u32> name_to_index;

private:
    uint8_t const* cursor;
    void read_token();
    void read_data();
    template<class T>
    T read_scalar() {
        assert(mapping);
        assert(mapping_size != 0);
        assert(cursor + sizeof(T) <= mapping + mapping_size);
        T ret;
        memcpy(&ret, (T*)cursor, sizeof(T));
        cursor += sizeof(T);
        return ret;
    }

    [[nodiscard]] bool is_open() const;
};

#endif //VULKAN_LLAMA_GGML_FILE_H
