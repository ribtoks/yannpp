#include "log.h"

#include <cstdio>
#include <stdarg.h>
#include <vector>

#include "common/array3d.h"
#include "common/shape.h"

namespace yannpp {
    void log(const char *fmt, ...) {
        va_list args;
        va_start(args, fmt);
        int result = vfprintf(stdout, fmt, args);
        fprintf(stdout, "\n");
    }

    void log_row(array3d_t<float> const &arr, int xi, int yj, std::vector<int> const &indices) {
        auto &data = arr.shape().data();
        int z_ = data[2], zn = indices.size();
        const bool zdots = z_ > zn;

        printf("[");
        for (int k = 0; k < zn - 1; k++) {
            int zk = indices[k];

            printf("%.6f, ", arr(xi, yj, zk));
        }

        if (zdots) { printf("... , "); }
        printf("%.6f", arr(xi, yj, zn - 1));
        printf("]");
    }

    void log_matrix(array3d_t<float> const &arr,
                    int xi,
                    std::vector<int> const &yindices,
                    std::vector<int> const &zindices) {
        auto &data = arr.shape().data();
        int y_ = data[1], yn = yindices.size();
        printf("[");
        bool ydots = y_ > yn;
        for (int j = 0; j < yn - 1; j++) {
            int yj = yindices[j];
            log_row(arr, xi, yj, zindices);
            printf(",\n    ");
        }
        if (ydots) { printf(" ..... \n    "); }
        log_row(arr, xi, yn - 1, zindices);
        printf("]");
    }

    void log(array3d_t<float> const &arr) {
        auto &shape = arr.shape();
        int x_ = shape.x(), y_ = shape.y(), z_ = shape.z();
        auto &data = shape.data();
        std::vector<int> indices[3];
        int nodots[3] = {4, 3, 6};
        for (size_t ii = 0; ii < 3; ii++) {
            bool fits = data[ii] <= nodots[ii];
            int maxn = fits ? data[ii] : nodots[ii];
            for (int jj = 0; jj < maxn; jj++) { indices[ii].push_back(jj); }
            if (!fits) { indices[ii].push_back(data[ii] - 1); }
        }

        printf("array3d_t(%d, %d, %d):\n", x_, y_, z_);

        const int xn = indices[0].size();
        const bool xdots = x_ > xn;
        printf("  [");
        for (int i = 0; i < xn - 1; i++) {
            int xi = indices[0][i];
            log_matrix(arr, xi, indices[1], indices[2]);
            printf(",\n\n   ");
        }
        if (xdots) { printf(" ..... \n\n   "); }
        log_matrix(arr, xn - 1, indices[1], indices[2]);

        printf("]\n");
    }
}
