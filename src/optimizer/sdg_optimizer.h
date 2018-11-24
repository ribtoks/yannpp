#ifndef SDG_ALGO_H
#define SDG_ALGO_H

#include "common/array3d.h"
#include "optimizer/optimizer.h"

namespace yannpp {
    template <typename T>
    class sdg_optimizer_t: public optimizer_t<T> {
    public:
        sdg_optimizer_t(size_t minibatch_size,
                        size_t input_size,
                        T decay_rate,
                        T learning_rate):
            minibatch_size_(minibatch_size),
            input_size_(input_size),
            weight_decay_(decay_rate),
            learning_rate_(learning_rate)
        {}

    public:
        virtual void update_bias(array3d_t<T> &b, array3d_t<T> &nabla_b) const override {
            // b = b - eta/minibatch_size * gradient_b
            T scale = learning_rate_ / (T)minibatch_size_;
            b.add(nabla_b.mul(-scale));
        }

        virtual void update_weights(array3d_t<T> &w, array3d_t<T> &nabla_w) const override {
            // w = w - eta/minibatch_size * gradient_w
            T scale = learning_rate_;
            T decay = T(1) - learning_rate_*weight_decay_ / (T)input_size_;
            w.mul(decay).add(nabla_w.mul(-scale));
        }

    private:
        size_t minibatch_size_;
        size_t input_size_;
        T weight_decay_;
        T learning_rate_;
    };
}

#endif // SDG_ALGO_H
