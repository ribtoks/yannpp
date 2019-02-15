#ifndef OPTIMIZATION_ALGORITHM_H
#define OPTIMIZATION_ALGORITHM_H

namespace yannpp {
    template <typename T>
    class array3d_t;

    template <typename T>
    class optimizer_t {
    public:
        virtual ~optimizer_t() {}
        virtual void update_bias(array3d_t<T> &b, array3d_t<T> &nabla_b) const = 0;
        virtual void update_weights(array3d_t<T> &w, array3d_t<T> &nabla_w) const = 0;
    };
}

#endif // OPTIMIZATION_ALGORITHM_H
