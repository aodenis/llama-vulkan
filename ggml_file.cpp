#include <fcntl.h>
#include <iostream>
#include <cstring>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cassert>
#include <utility>
#include "ggml_file.h"
#include "types.h"

using namespace std;

// #define LLAMA_CPP_ISO

ggml_file::ggml_file(const char *filepath) : header{} {
    int fd = open(filepath, O_RDONLY);
    if (fd < 0) {
        cerr << "Cannot open " << filepath << ": " << strerror(errno) << endl;
        return;
    }
    struct stat statbuf{};
    if(fstat(fd, &statbuf) < 0) {
        cerr << "Cannot stat " << filepath << ": " << strerror(errno) << endl;
        if (close(fd) < 0) {
            cerr << "Additionally, close on its failed too: " << strerror(errno) << endl;
        }
        return;
    }
    mapping_size = statbuf.st_size;
    mapping = (uint8_t*)mmap(nullptr, mapping_size, PROT_READ, MAP_SHARED, fd, 0);
    if(mapping == MAP_FAILED) {
        mapping_size = 0;
        mapping = nullptr;
        cerr << "Cannot mmap " << filepath << ": " << strerror(errno) << endl;
        if (close(fd) < 0) {
            cerr << "Additionally, close on its failed too: " << strerror(errno) << endl;
        }
        return;
    }
    if (close(fd) < 0) {
        cerr << "close() failed on ggml file's fd, strange: " << strerror(errno) << endl;
    }

    // Parsing the file
    assert(mapping_size > sizeof(ggml_header));
    memcpy(&header, mapping, sizeof(ggml_header));
    assert(header.magic == 0x67676a74);
    assert(header.file_version == 1);
    assert(header.vocab_size > 0);
    assert(header.n_heads > 0);
    assert(header.n_layers > 0);
    assert(header.rot * header.n_heads == header.dim);
    assert(header.ftype <= 3);
    assert(header.ftype >= 0);

    cursor = mapping + sizeof(ggml_header);
    for(int32_t i = 0; i < header.vocab_size; ++i) {
        read_token();
    }
    // Done
    while(cursor != mapping + mapping_size) {
        read_data();
    }

    ff_size = get_buffer_descriptor("layers.0.feed_forward.w1").shape1;
}

void ggml_file::read_token() {
    auto token_size = read_scalar<int32_t>();
    assert(token_size >= 0);
    assert(mapping + token_size <= mapping + mapping_size);

    ggml_token& token = tokens.emplace_back();
    token.text.resize(token_size);
    if (token_size) {
        ::memcpy(token.text.data(), cursor, token_size);
    }
    cursor += token_size;

    token.score = read_scalar<float>();
}

void ggml_file::read_data() {
    auto n_dims = read_scalar<u32>();
    auto name_len = read_scalar<u32>();
    auto ftype = read_scalar<u32>();
    assert(ftype <= 3);
    assert(n_dims > 0);
    assert(n_dims < 3);
    auto shape1 = read_scalar<u32>();
    auto shape2 = (n_dims == 2) ? read_scalar<u32>() : 1;
    if (n_dims == 2) {
        ::swap(shape1, shape2);
    }
    vector<char> name;
    name.reserve(name_len + 1);
    name.resize(name_len);
    if (name_len) {
        ::memcpy(name.data(), cursor, name_len);
    }
    cursor += name_len;
    for(char c : name) {
        assert(c);
    }
    name.emplace_back(0);
    size_t offset = (cursor - mapping);
    offset = (offset + 31) & (~31UL);
    auto& new_table = tables.emplace_back(string(name.data()), static_cast<ggml_value_type>(ftype), offset, shape1, shape2);
    assert(name_to_index.emplace(new_table.name, tables.size() - 1).second);
    cursor = mapping + new_table.offset + new_table.size;
}

ggml_file::~ggml_file() {
    if(mapping) {
        munmap(mapping, mapping_size);
        mapping = nullptr;
        mapping_size = 0;
    }
}

void ggml_file::print_info() const {
    assert(mapping != nullptr);
    u32 magic_copy[2] = {header.magic, 0};
    cout << "       magic: " << ((char*)magic_copy) << "\n";
    cout << "file_version: " << header.file_version << "\n";
    cout << "  vocab_size: " << header.vocab_size << "\n";
    cout << "         dim: " << header.dim << "\n";
    cout << " multiple_of: " << header.multiple_of << "\n";
    cout << "     n_heads: " << header.n_heads << "\n";
    cout << "    n_layers: " << header.n_layers << "\n";
    cout << "per head dim: " << header.rot << "\n";
    cout << "       ftype: " << header.ftype << "\n";
    cout << "fast_forward: " << ff_size << endl;
}

ggml_data_descriptor::ggml_data_descriptor(std::string _name,
                                           ggml_value_type _ftype,
                                           size_t _offset,
                                           int32_t _shape1,
                                           int32_t _shape2) : name(std::move(_name)),
                                                              ftype(_ftype),
                                                              offset(_offset),
                                                              shape1(_shape1),
                                                              shape2(_shape2),
                                                              size(size_in_file()) {

}

size_t ggml_data_descriptor::size_in_file() const {
    if (ftype == ggml_value_type::f16) {
        return shape1 * shape2 * 2;
    }
    if (ftype == ggml_value_type::f32) {
        return shape1 * shape2 * 4;
    }
    size_t base_size = shape1 * shape2 * ((ftype == ggml_value_type::q4_0) ? 20 : 24);
    assert((base_size % 32) == 0);
    return base_size / 32;
}

size_t ggml_data_descriptor::size_for_type(ggml_value_type _ftype) const {
    if (_ftype == ggml_value_type::f16) {
        return shape1 * shape2 * 2;
    }
    if (_ftype == ggml_value_type::f32) {
        return shape1 * shape2 * 4;
    }
    size_t base_size = shape1 * shape2 * ((_ftype == ggml_value_type::q4_0) ? 20 : 24);
    assert((base_size % 32) == 0);
    return base_size / 32;
}

// The following function were copied from llama.cpp source
void ggml_file::try_add_bigram(int left, int right) {
    if ((left == -1) or (right == -1)) {
        return;
    }

    size_t seek_size = symbols_.at(left).n + symbols_.at(right).n;
    char const* seek_addr = symbols_.at(left).text;
    for (u32 j = 0; j < tokens.size(); ++j) {
#ifdef LLAMA_CPP_ISO
            u32 k = tokens.size() - 1 - j;
#else
            u32 k = j;
#endif
        auto& token = tokens.at(k);
        if (token.text.size() != seek_size) {
            continue;
        }
        if (memcmp(token.text.data(), seek_addr, seek_size) != 0) {
            continue;
        }

        // Found
        work_queue_.push({
            .left = left,
            .right = right,
            .score = token.score,
            .size = seek_size
        });
        return;
    }
}

static size_t utf8_len(char src) {
    const size_t lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    return lookup[static_cast<uint8_t>(src) >> 4];
}

void ggml_file::tokenize(std::vector<uint32_t> &output, const std::string& text, bool bos) {
    if (text.empty()) {
        output.clear();
        return;
    }

    if (bos) {
        output.push_back(1);
    }

    // split string into utf8 chars
    for (size_t offs = 0; offs < text.size(); offs += symbols_.back().n) {
        llama_sp_symbol& sym = symbols_.emplace_back();
        size_t char_len = std::min(text.size() - offs, utf8_len(text[offs]));
        sym.text = text.c_str() + offs;
        sym.n = char_len;
        sym.next = static_cast<int>(symbols_.size());
        sym.prev = sym.next - 2;
    }

    assert(not symbols_.empty());
    symbols_.back().next = -1;

    // seed the work queue with all possible 2-character tokens.
    for (int i = 1; i < symbols_.size(); ++i) {
        try_add_bigram(i - 1, i);
    }

    // keep substituting the highest frequency pairs for as long as we can.
    while (not work_queue_.empty()) {
        auto bigram = work_queue_.top();
        work_queue_.pop();

        auto& left_sym = symbols_.at(bigram.left);
        auto& right_sym = symbols_.at(bigram.right);

        // if one of the symbols already got merged, skip it.
        if (left_sym.n == 0 || right_sym.n == 0 ||
            left_sym.n + right_sym.n != bigram.size) {
            continue;
        }

        // merge the right sym into the left one
        left_sym.n += right_sym.n;
        right_sym.n = 0;

        // remove the right sym from the chain
        left_sym.next = right_sym.next;
        if (right_sym.next >= 0) {
            symbols_[right_sym.next].prev = bigram.left;
        }

        // find more substitutions
        try_add_bigram(left_sym.prev, bigram.left);
        try_add_bigram(bigram.left, left_sym.next);
    }

    for (int i = 0; i != -1; i = symbols_[i].next) {
        auto & symbol = symbols_[i];
        bool found = false;
        for (uint32_t j = 0; j < tokens.size(); ++j) {
#ifdef LLAMA_CPP_ISO
            u32 k = tokens.size() - 1 - j;
#else
            u32 k = j;
#endif
            auto& token = tokens.at(k);
            if (token.text.size() != symbol.n) {
                continue;
            }
            if (memcmp(token.text.data(), symbol.text, symbol.n) != 0) {
                continue;
            }
            found = true;
            output.push_back(k);
            break;
        }

        if (not found) {
            // output any symbols that did not form tokens as bytes.
            for (int j = 0; j < (int) symbol.n; ++j) {
                uint32_t token_id = static_cast<uint8_t>(symbol.text[j]) + 3;
                output.push_back(token_id);
            }
        }
    }
}

std::vector<ggml_data_descriptor> const &ggml_file::get_buffers() {
    return tables;
}

ggml_data_descriptor const &ggml_file::get_buffer_descriptor(const string &s) const {
    if (auto it = name_to_index.find(s); it != name_to_index.end()) {
        return tables.at(it->second);
    }
    auto jt = name_to_index.find(s + ".weight");
    assert(jt != name_to_index.end());
    return tables.at(jt->second);
}
