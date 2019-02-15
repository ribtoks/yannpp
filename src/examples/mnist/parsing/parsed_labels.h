#ifndef PARSED_LABELS_H
#define PARSED_LABELS_H

#include <vector>
#include <cstdint>
#include <string>
#include <fstream>

namespace yannpp {
    class parsed_labels_t {
    public:
        using labels = std::vector<uint8_t>;

    public:
        parsed_labels_t(const std::string &labels);

    public:
        size_t size() const { return labels_count_; }

    public:
        labels::iterator begin() { return labels_.begin(); }
        labels::iterator end() { return labels_.end(); }

    private:
        void read_header();
        void read_labels();

    private:
        std::ifstream stream_;
        labels labels_;
        size_t labels_count_ = 0;
    };
}

#endif // PARSED_LABELS_H
