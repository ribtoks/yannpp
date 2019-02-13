#include "log.h"

#include <algorithm>
#include <cstdio>
#include <stdarg.h>
#include <vector>
#include <utility>

#include "common/array3d.h"
#include "common/shape.h"

namespace yannpp {
    void log(const char *fmt, ...) {
        va_list args;
        va_start(args, fmt);
        int result = vfprintf(stdout, fmt, args);
        fprintf(stdout, "\n");
    }

    bool segmentsOverlap(const std::pair<int, int> &a, const std::pair<int, int> &b) {
        if (a.first <= b.first) {
            return (b.first <= a.second) || (b.first == a.second+1);
        } else {
            return (a.first <= b.second) || (a.first == b.second+1);
        }
    }

    std::pair<int, int> unionOverlappingSegments(const std::pair<int, int> &a, const std::pair<int, int> &b) {
        return std::make_pair(std::min(a.first, b.first), std::max(a.second, b.second));
    }

    void log_zchunk(array3d_t<float> const &arr, int x, int y, std::pair<int, int> const &zrange) {
        assert(zrange.first <= zrange.second);
        for (int z = zrange.first; z < zrange.second; z++) {
            printf("%.6f, ", arr(x, y, z));
        }

        printf("%.6f", arr(x, y, zrange.second));
    }

    void log_row(array3d_t<float> const &arr, int x, int y) {
        auto &data = arr.shape().data();
        int z = arr.shape().z();
        std::pair<int, int> start(0, std::min(std::max(z-2, 0), 3));
        std::pair<int, int> end(std::max(0, z-2), z-1);

        printf("[");

        if (segmentsOverlap(start, end)) {
            auto all = unionOverlappingSegments(start, end);
            log_zchunk(arr, x, y, all);
        } else {
            log_zchunk(arr, x, y, start);
            printf(", ... , ");
            log_zchunk(arr, x, y, end);
        }

        printf("]");
    }

    void log_ychunk(array3d_t<float> const &arr, int x, std::pair<int, int> const &yrange) {
        for (int y = yrange.first; y < yrange.second; y++) {
            log_row(arr, x, y);
            printf(",\n    ");
        }

        log_row(arr, x, yrange.second);
    }

    void log_matrix(array3d_t<float> const &arr, int x) {
        int y = arr.shape().y();
        std::pair<int, int> start(0, std::min(std::max(y-2, 0), 3));
        std::pair<int, int> end(std::max(0, y-2), y-1);

        printf("[");

        if (segmentsOverlap(start, end)) {
            auto all = unionOverlappingSegments(start, end);
            log_ychunk(arr, x, all);
        } else {
            log_ychunk(arr, x, start);
            printf(",\n");
            printf("     ..... \n    ");
            log_ychunk(arr, x, end);
        }

        printf("]");
    }

    void log_xchunk(array3d_t<float> const &arr, std::pair<int, int> const &xrange) {
        for (int x = xrange.first; x < xrange.second; x++) {
            log_matrix(arr, x);
            printf(",\n\n   ");
        }

        log_matrix(arr, xrange.second);
    }

    void log(array3d_t<float> const &arr) {
        printf("array3d_t(%d, %d, %d):\n", arr.shape().x(), arr.shape().y(), arr.shape().z());
        int x = arr.shape().x();
        std::pair<int, int> start(0, std::min(std::max(x-2, 0), 3));
        std::pair<int, int> end(std::max(0, x-2), x-1);

        printf("  [");

        if (segmentsOverlap(start, end)) {
            auto all = unionOverlappingSegments(start, end);
            log_xchunk(arr, all);
        } else {
            log_xchunk(arr, start);
            printf(",\n\n");
            printf("    ..... \n\n   ");
            log_xchunk(arr, end);
        }

        printf("]\n");
    }
}
