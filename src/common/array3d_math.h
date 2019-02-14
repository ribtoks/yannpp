#ifndef ARRAY3D_MATH_H
#define ARRAY3D_MATH_H

#include <exception>
#include <cmath>

#include "common/array3d.h"
#include "common/shape.h"

namespace yannpp {
    template<typename T>
    T sigmoid(T x) {
        return T(1.0)/(T(1.0) + exp(-x));
    }

    template<typename T>
    T sigmoid_derivative(T x) {
        T sigmoid_x = sigmoid(x);
        return sigmoid_x * (T(1.0) - sigmoid_x);
    }

    template<typename T>
    T relu(T x) {
        return x < T(0.0) ? T(0.0) : x;
    }

    template<typename T>
    array3d_t<T> sigmoid_v(array3d_t<T> const &x) {
        array3d_t<T> result(x);
        result.apply(sigmoid<T>);
        return result;
    }

    template<typename T>
    array3d_t<T> sigmoid_derivative_v(array3d_t<T> const &x) {
        array3d_t<T> result(x);
        result.apply(sigmoid_derivative<T>);
        return result;
    }

    template<typename T>
    array3d_t<T> stable_softmax_v(array3d_t<T> const &x) {
        array3d_t<T> result(x);
        const int size = (int)result.size();
        T x_max = x.max();

        T sum = 0.0;
        for (int i = 0; i < size; i++) {
            T fi = exp(x(i) - x_max);;
            result(i) = fi;
            sum += fi;
        }

        for (int i = 0; i < size; i++) {
            result(i) /= sum;
        }

        return result;
    }

    template<typename T>
    array3d_t<T> relu_v(array3d_t<T> const &x) {
        array3d_t<T> result(x);
        result.apply(relu<T>);
        return result;
    }

    template<typename T>
    size_t argmax1d(array3d_t<T> const &v) {
        assert(v.size() > 0);
        assert(v.shape().dim() == 1);

        const size_t size = v.size();
        T max_v = v(0);
        size_t max_i = 0;
        for (size_t i = 1; i < size; i++) {
            T vi = v(i);
            if (vi > max_v) {
                max_v = vi;
                max_i = i;
            }
        }
        return max_i;
    }

    template<typename T>
    T inner_product(array3d_t<T> const &a, array3d_t<T> const &b) {
        assert(a.shape().dim() == b.shape().dim());
        assert(a.size() == b.size());

        T sum = 0;
        auto &a_raw = a.data();
        auto &b_raw = b.data();
        const size_t size = a_raw.size();
        for (size_t i = 0; i < size; i++) {
            sum += a_raw[i] * b_raw[i];
        }

        return sum;
    }

    // dot product of two slices (used in convolutions)
    template<typename T>
    T dot(typename array3d_t<T>::slice3d const &a, typename array3d_t<T>::slice3d const &b) {
        assert(a.shape() == b.shape());

        T sum = 0;
        auto it_a = a.iterator();
        auto it_b = b.iterator();
        for (; it_a.is_valid() && it_b.is_valid(); ++it_a, ++it_b) {
            sum += a.at(it_a) * b.at(it_b);
        }

        return sum;
    }

    // dot product of matrix (H, W, 1) and vector (W, 1, 1)
    // result is vector of size (H, 1, 1)
    template<typename T>
    array3d_t<T> dot21(array3d_t<T> const &m, array3d_t<T> const &v) {
        assert(m.shape().dim() == 2);
        assert(v.shape().dim() == 1);
        assert(m.shape().y() == v.shape().x());

        const size_t height = m.shape().x();
        const size_t width = m.shape().y();
        array3d_t<T> result(shape_row(height), 0);
        for (size_t i = 0; i < height; i++) {
            T sum = 0;
            for (size_t j = 0; j < width; j++) {
                sum += v(j) * m(i, j);
            }
            result(i) = sum;
        }
        return result;
    }

    // outer product of vectors (H, 1, 1) and (W, 1, 1)
    // is matrix (H, W, 1)
    template<typename T>
    array3d_t<T> outer_product(array3d_t<T> const &a, array3d_t<T> const &b) {
        assert(a.shape().dim() == b.shape().dim());
        assert(a.shape().dim() == 1);

        const size_t height = a.shape().x();
        const size_t width = b.shape().x();

        array3d_t<T> c(shape3d_t(height, width, 1), 0);

        for (size_t i = 0; i < height; i++) {
            for (size_t j = 0; j < width; j++) {
                c(i, j) = a(i) * b(j);
            }
        }

        return c;
    }

    // dot product of matrix (H, W, 1) and vector (H, 1, 1) columnwise
    // result is vector (W, 1, 1)
    template<typename T>
    array3d_t<T> transpose_dot21(array3d_t<T> const &m, array3d_t<T> const &v) {
        assert(m.shape().dim() == 2);
        assert(v.shape().dim() == 1);
        assert(m.shape().x() == v.shape().x());

        const size_t width = m.shape().y();
        const size_t height = m.shape().x();
        array3d_t<T> output(shape_row(width), 0);

        for (size_t j = 0; j < width; j++) {
            T sum = 0;
            for (size_t i = 0; i < height; i++) {
                sum += m(i, j) * v(i);
            }
            output(j) = sum;
        }

        return output;
    }
}

#endif // ARRAY3D_MATH_H
