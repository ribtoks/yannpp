#ifndef ARRAY3D_MATH_H
#define ARRAY3D_MATH_H

#include "common/array3d.h"
#include <exception>

namespace yannpp {
    double sigmoid(double x);
    double sigmoid_derivative(double x);
    double relu(double x);
    double relu_derivative(double x);
    array3d_t<double> sigmoid_v(array3d_t<double> const &x);
    array3d_t<double> sigmoid_derivative_v(array3d_t<double> const &x);
    array3d_t<double> stable_softmax_v(array3d_t<double> const &x);
    array3d_t<double> relu_v(array3d_t<double> const &x);

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

    template<typename T>
    array3d_t<T> dot21(array3d_t<T> const &m, array3d_t<T> const &v) {
        assert(m.shape().dim() == 2);
        assert(v.shape().dim() == 1);
        assert(m.shape().x() == v.shape().x());

        const size_t height = m.shape().y();
        const size_t width = m.shape().x();
        array3d_t<T> result(shape_row(height), 0);
        for (size_t i = 0; i < height; i++) {
            T sum = 0;
            for (size_t j = 0; j < width; j++) {
                sum += v(j) * m(j, i);
            }
            result(i) = sum;
        }
        return result;
    }

    template<typename T>
    array3d_t<T> outer_product(array3d_t<T> const &a, array3d_t<T> const &b) {
        assert(a.shape().dim() == b.shape().dim());
        assert(a.shape().dim() == 1);

        const size_t height = a.shape().x();
        const size_t width = b.shape().x();

        array3d_t<T> c(shape_matrix(height, width), 0);

        for (size_t i = 0; i < height; i++) {
            for (size_t j = 0; j < width; j++) {
                c(j, i) = a(i) * b(j);
            }
        }

        return c;
    }

    template<typename T>
    array3d_t<T> transpose_dot21(array3d_t<T> const &m, array3d_t<T> const &v) {
        assert(m.shape().dim() == 2);
        assert(v.shape().dim() == 1);
        assert(m.shape().y() == v.shape().x());

        const size_t width = m.shape().x();
        const size_t height = m.shape().y();
        array3d_t<T> output(shape_row(width), 0);

        for (size_t i = 0; i < width; i++) {
            T sum = 0;
            for (size_t j = 0; j < height; j++) {
                sum += m(i, j) * v(j);
            }
            output(i) = sum;
        }

        return output;
    }
}

#endif // ARRAY3D_MATH_H
