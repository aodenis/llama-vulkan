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
#include "utils.h"

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
    assert(header.vocab_size > 0);
    assert(header.n_heads > 0);
    assert(header.n_layers > 0);
    assert(header.rot * header.n_heads == header.dim);

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
    token.text = string((const char*)cursor, token_size);
    cursor += token_size;

    token.score = read_scalar<float>();
}

void ggml_file::read_data() {
    size_t cursor_save = cursor - mapping;
    auto n_dims = read_scalar<u32>();
    auto name_len = read_scalar<u32>();
    auto ftype = read_scalar<u32>();
    assert(n_dims > 0);
    assert(n_dims < 3);
    u32 shape1 = read_scalar<u32>();
    u32 shape2 = 1;
    if (n_dims == 2) {
        shape2 = shape1;
        shape1 = read_scalar<u32>();
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
    if (not ((ftype <= 3) or (ftype == 8))) {
        cerr << "Bad format for matrix in input file" << endl;
        const char* type_name = ftype_name(static_cast<ggml_value_type>(ftype));
        cerr << "Unsupported type ";
        if (type_name) {
            cerr << type_name;
        } else {
            cerr << ftype;
        }
        cerr << " for table " << name.data() << " at offset 0x" << hex << cursor_save << dec << endl;
        exit(1);
    }

    size_t offset = (cursor - mapping);
    offset = (offset + 31) & (~31UL);
    auto& new_table = tables.emplace_back(string(name.data()), static_cast<ggml_value_type>(ftype), 0+header.file_version, offset, shape1, shape2);
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
                                           u32 _model_version,
                                           size_t _offset,
                                           int32_t _shape1,
                                           int32_t _shape2) : name(std::move(_name)),
                                                              ftype(_ftype),
                                                              model_version(_model_version),
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
    assert(((shape1 * shape2) % 32) == 0); // This condition is slightly too restrictive, to be changed if this fails
    size_t sz = shape1 * shape2;
    if (ftype == ggml_value_type::q4_0) {
        if (model_version == 1) {
            return (sz * 20) / 32;
        } else {
            return (sz * 18) / 32;
        }
    } else if (ftype == ggml_value_type::q8_0) {
        if (model_version == 1) {
            return (sz * 36) / 32;
        } else {
            return (sz * 34) / 32;
        }
    }

    assert (false);
}

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

void try_add_bigram(int left, int right, vector<llama_sp_symbol>& symbols, llama_sp_bigram::queue& work_queue, vector<ggml_token> const& tokens) {
    if ((left == -1) or (right == -1)) {
        return;
    }

    size_t seek_size = symbols.at(left).n + symbols.at(right).n;
    char const* seek_addr = symbols.at(left).text;
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
        if (memcmp(token.text.c_str(), seek_addr, seek_size) != 0) {
            continue;
        }

        // Found
        work_queue.push({
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

void ggml_file::tokenize(std::vector<uint32_t> &output, const std::string& text, bool bos) const {
    if (text.empty()) {
        return;
    }

    vector<llama_sp_symbol> symbols;
    llama_sp_bigram::queue work_queue;

    if (bos) {
        output.push_back(1);
    }

    // split string into utf8 chars
    for (size_t offs = 0; offs < text.size(); offs += symbols.back().n) {
        llama_sp_symbol& sym = symbols.emplace_back();
        size_t char_len = std::min(text.size() - offs, utf8_len(text[offs]));
        sym.text = text.c_str() + offs;
        sym.n = char_len;
        sym.next = static_cast<int>(symbols.size());
        sym.prev = sym.next - 2;
    }

    assert(not symbols.empty());
    symbols.back().next = -1;

    // seed the work queue with all possible 2-character tokens.
    for (int i = 1; i < symbols.size(); ++i) {
        try_add_bigram(i - 1, i, symbols, work_queue, tokens);
    }

    // keep substituting the highest frequency pairs for as long as we can.
    while (not work_queue.empty()) {
        auto bigram = work_queue.top();
        work_queue.pop();

        auto& left_sym = symbols.at(bigram.left);
        auto& right_sym = symbols.at(bigram.right);

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
            symbols[right_sym.next].prev = bigram.left;
        }

        // find more substitutions
        try_add_bigram(left_sym.prev, bigram.left, symbols, work_queue, tokens);
        try_add_bigram(bigram.left, left_sym.next, symbols, work_queue, tokens);
    }

    for (int i = 0; i != -1; i = symbols[i].next) {
        auto & symbol = symbols[i];
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
            if (memcmp(token.text.c_str(), symbol.text, symbol.n) != 0) {
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

ggml_data_descriptor const &ggml_file::get_buffer_descriptor(const string &s) const {
    if (auto it = name_to_index.find(s); it != name_to_index.end()) {
        return tables.at(it->second);
    }
    auto jt = name_to_index.find(s + ".weight");
    assert(jt != name_to_index.end());
    return tables.at(jt->second);
}

string ggml_file::tokens_to_text(const u32 * _tokens, u32 count) const {
    u32 sz = 0;
    for(u32 j = 0; j < count; ++j) {
        sz += tokens.at(_tokens[j]).text.size();
    }
    string res;
    res.reserve(sz + 1);
    for(u32 j = 0; j < count; ++j) {
        res += tokens.at(_tokens[j]).text;
    }
    return res;
}

std::vector<ggml_token> const &ggml_file::get_tokens() const {
    return tokens;
}

bool ggml_file::is_open() const {
    return (mapping != nullptr) and (mapping != MAP_FAILED);
}
