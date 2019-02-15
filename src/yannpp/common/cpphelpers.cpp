#include "cpphelpers.h"

#include <algorithm>
#include <numeric>

namespace yannpp {
    std::vector<std::vector<size_t>> batch_indices(size_t size, size_t batch_size) {
        std::vector<size_t> indices(size, 0);
        std::iota(indices.begin(), indices.end(), 0);
        std::random_shuffle(indices.begin(), indices.end());

        std::vector<std::vector<size_t>> batches;
        for(size_t i = 0; i < size; i += batch_size) {
            auto last = std::min(size, i + batch_size);
            batches.emplace_back(indices.begin() + i, indices.begin() + last);
        }

        return batches;
    }
}
