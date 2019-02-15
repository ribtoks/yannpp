#ifndef ACTIVATOR_H
#define ACTIVATOR_H

#include <functional>

#include <yannpp/common/array3d.h>

namespace yannpp {
    template<typename T>
    class activator_t {
        using activator_func_t = std::function<array3d_t<T>(const array3d_t<T>&)>;
    public:
        activator_t(activator_func_t const &activation_func,
                    activator_func_t const &derivative):
            activation_func_(activation_func),
            derivative_(derivative)
        { }

        activator_t(activator_t const &other):
            activation_func_(other.activation_func_),
            derivative_(other.derivative_)
        { }

    public:
        array3d_t<T> activate(array3d_t<T> const &v) const { return activation_func_(v); }
        array3d_t<T> derivative(array3d_t<T> const &v) const { return derivative_(v); }

    private:
        activator_func_t activation_func_;
        activator_func_t derivative_;
    };
}

#endif // ACTIVATOR_H
