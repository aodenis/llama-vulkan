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
    std::vector<char> text;
    float score;
};

class ggml_data_descriptor {
public:
    ggml_data_descriptor(std::string name, ggml_value_type ftype, size_t offset, int32_t shape1, int32_t shape2 = 1);
    const std::string name;
    const ggml_value_type ftype;
    const size_t offset;
    const u32 shape1;
    const u32 shape2;
    const size_t size;

    [[nodiscard]] size_t size_in_file() const;
    [[nodiscard]] size_t size_for_type(ggml_value_type ftype) const;
};

struct llama_sp_symbol {
    using index = int;
    index prev;
    index next;
    const char * text;
    size_t n;
};

struct llama_sp_bigram {
    struct comparator {
        bool operator()(llama_sp_bigram & l, llama_sp_bigram & r) {
            return (l.score < r.score) || (l.score == r.score && l.left > r.left);
        }
    };
    using queue_storage = std::vector<llama_sp_bigram>;
    using queue = std::priority_queue<llama_sp_bigram, queue_storage, comparator>;
    llama_sp_symbol::index left;
    llama_sp_symbol::index right;
    float score;
    size_t size;
};

class ggml_file {
    friend class llava_context;
    friend class llava_buffer;
    friend class llava_layer;
public:
    ggml_file(const char* filepath);
    ~ggml_file();
    void print_info() const;
    void tokenize(std::vector<u32> &output, const std::string &text, bool bos);
    std::vector<ggml_data_descriptor> const& get_buffers();
    ggml_data_descriptor const& get_buffer_descriptor(const string& s) const;

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
        T ret = *((T*)cursor);
        cursor += sizeof(T);
        return ret;
    }

private: // Tokenizer
    std::vector<llama_sp_symbol> symbols_;
    llama_sp_bigram::queue work_queue_;
    void try_add_bigram(int left, int right);
};

#endif //VULKAN_LLAMA_GGML_FILE_H
