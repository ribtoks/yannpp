#include "parsed_labels.h"
#include <exception>
#include "common/cpphelpers.h"

#define MAGIC_NUMBER 0x00000801
#define LABELS_COUNT 10000

namespace yannpp {
    parsed_labels_t::parsed_labels_t(const std::string& filepath):
        stream_(filepath, std::ios::in | std::ios::binary)
    {
        if (!stream_) {
            throw std::runtime_error(string_format("Cannot open file %s", filepath.c_str()));
        }

        read_header();
        read_labels();
    }

    void parsed_labels_t::read_header() {
        uint32_t magic_number;
        if (stream_.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number))) {
            magic_number = swap_endian<uint32_t>(magic_number);
            if (magic_number != MAGIC_NUMBER) {
                throw std::runtime_error("Magic number does not match");
            }
        }

        uint32_t labels_number;
        if (stream_.read(reinterpret_cast<char*>(&labels_number), sizeof(labels_number))) {
            labels_number = swap_endian<uint32_t>(labels_number);
            if (labels_number % LABELS_COUNT != 0) {
                throw std::runtime_error(string_format("Labels number does not match: %d found", labels_number));
            }
        }
        labels_count_ = labels_number;
    }

    void parsed_labels_t::read_labels() {
        std::vector<uint8_t> data;
        data.resize(labels_count_);

        if (stream_.read(reinterpret_cast<char*>(&data[0]), data.size())) {
            labels_.swap(data);
        }
    }
}
