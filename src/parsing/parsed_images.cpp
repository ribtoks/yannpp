#include "parsed_images.h"
#include <cstdint>
#include <exception>
#include "common/cpphelpers.h"

#define MAGIC_NUMBER 0x00000803
#define IMAGES_COUNT 10000
#define ROWS_NUMBER 28
#define COLUMNS_NUMBER 28

namespace yannpp {
    parsed_images_t::iterator::iterator(std::ifstream &stream, size_t columns, size_t rows, size_t index):
        stream_(stream),
        columns_(columns),
        rows_(rows),
        local_index_(0),
        global_index_(index)
    {
    }

    void parsed_images_t::iterator::refill_cache() {
        std::vector<image_data> cache;
        image_data data;
        data.resize(columns_ * rows_);

        int count = 600;
        while (count--) {
            if (stream_.read(reinterpret_cast<char*>(&data[0]), data.size())) {
                cache.push_back(data);
            } else {
                break;
            }
        }

        cache_.swap(cache);
    }

    const parsed_images_t::image_data &parsed_images_t::iterator::operator*() {
        if (local_index_ >= cache_.size()) {
            refill_cache();
            if (cache_.size() > 0) {
                local_index_ = 0;
            }
        }

        if (local_index_ < cache_.size()) {
            auto &result = cache_[local_index_];
            return result;
        } else {
            throw std::logic_error("Cannot read more images");
        }
    }

    bool parsed_images_t::iterator::operator==(const iterator &other) {
        return global_index_ == other.global_index_;
    }

    bool parsed_images_t::iterator::operator!=(const iterator &other) {
        return global_index_ != other.global_index_;
    }

    parsed_images_t::parsed_images_t(const std::string &filepath):
        stream_(filepath, std::ios::in | std::ios::binary)
    {
        if (!stream_) {
            throw std::runtime_error(string_format("Cannot open file %s", filepath.c_str()));
        }

        read_header();
    }

    void parsed_images_t::read_header() {
        uint32_t magic_number;
        if (stream_.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number))) {
            magic_number = swap_endian<uint32_t>(magic_number);
            if (magic_number != MAGIC_NUMBER) {
                throw std::runtime_error("Magic number does not match");
            }
        }

        uint32_t images_number;
        if (stream_.read(reinterpret_cast<char*>(&images_number), sizeof(images_number))) {
            images_number = swap_endian<uint32_t>(images_number);
            if (images_number % IMAGES_COUNT != 0) {
                throw std::runtime_error(string_format("Images number does not match: %d found", images_number));
            }
        }
        images_count_ = images_number;

        uint32_t rows_number;
        if (stream_.read(reinterpret_cast<char*>(&rows_number), sizeof(rows_number))) {
            rows_number = swap_endian<uint32_t>(rows_number);
            if (rows_number != ROWS_NUMBER) {
                throw std::runtime_error("Rows number does not match");
            }
        }
        rows_ = rows_number;

        uint32_t columns_number;
        if (stream_.read(reinterpret_cast<char*>(&columns_number), sizeof(columns_number))) {
            columns_number = swap_endian<uint32_t>(columns_number);
            if (columns_number != COLUMNS_NUMBER) {
                throw std::runtime_error("Columns number does not match");
            }
        }
        columns_ = columns_number;
    }

    parsed_images_t::iterator parsed_images_t::begin() {
        return parsed_images_t::iterator(stream_, columns_, rows_, 0);
    }

    parsed_images_t::iterator parsed_images_t::end() {
        return parsed_images_t::iterator(stream_, columns_, rows_, images_count_);
    }
}
