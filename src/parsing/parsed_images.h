#ifndef IMAGES_PARSER_H
#define IMAGES_PARSER_H

#include <string>
#include <fstream>
#include <vector>

namespace yannpp {
    class parsed_images_t {
    public:
        using image_data = std::vector<uint8_t>;

    public:
        class iterator {
        public:
            iterator(std::ifstream &stream, size_t columns, size_t rows, size_t index=0);

        private:
            void refill_cache();

        public:
            const image_data& operator*();
            iterator& operator++() { local_index_++; global_index_++; return *this; }
            bool operator==(const iterator &other);
            bool operator!=(const iterator &other);

        private:
            std::ifstream &stream_;
            size_t columns_;
            size_t rows_;
            size_t local_index_;
            size_t global_index_;
            std::vector<image_data> cache_;
        };

    public:
        parsed_images_t(const std::string &filepath);

    public:
        size_t size() const { return images_count_; }
        size_t img_width() const { return columns_; }

    public:
        iterator begin();
        iterator end();

    private:
        void read_header();

    private:
        std::ifstream stream_;
        size_t images_count_ = 0;
        size_t rows_ = 0;
        size_t columns_ = 0;
    };
}

#endif // IMAGES_PARSER_H










