#ifndef VULKAN_LLAMA_GGML_FILE_H
#define VULKAN_LLAMA_GGML_FILE_H

#include <cstdint>
#include <vector>
#include <queue>
#include <string>
#include <cassert>

#define GGML_TYPE_F32 0
#define GGML_TYPE_F16 1
#define GGML_TYPE_Q4_0 2
#define GGML_TYPE_Q4_1 3

struct ggml_header {
    int32_t magic;
    int32_t file_version;
    int32_t vocab_size;
    int32_t dim;
    int32_t multiple_of;
    int32_t n_heads;
    int32_t n_layers;
    int32_t rot;
    int32_t ftype;
} __attribute__((__packed__));

struct ggml_token {
    std::vector<char> text;
    float score;
};

class ggml_data_descriptor {
public:
    ggml_data_descriptor(std::string name, int32_t ftype, size_t offset, int32_t shape1, int32_t shape2 = 1);
    const std::string name;
    const uint32_t ftype;
    const size_t offset;
    const uint32_t shape1;
    const uint32_t shape2;
    const size_t size;

    [[nodiscard]] size_t size_in_file() const;
    [[nodiscard]] size_t size_for_type(uint32_t ftype) const;
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
public:
    ggml_file(const char* filepath);
    ~ggml_file();
    void print_info() const;
    void tokenize(std::vector<uint32_t> &output, const std::string &text, bool bos);
    std::vector<ggml_data_descriptor> const& get_buffers();

private:
    uint8_t* mapping = nullptr;
    std::size_t mapping_size = 0;

private:
    ggml_header header;
    std::vector<ggml_token> tokens;
    std::vector<ggml_data_descriptor> tables;

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
