#include "array3d_math.h"

#include <cmath>

namespace yannpp {
    double sigmoid(double x) {
        return 1.0/(1.0 + exp(-x));
    }

    double sigmoid_derivative(double x) {
        double sigmoid_x = sigmoid(x);
        return sigmoid_x * (1.0 - sigmoid_x);
    }

    double relu(double x) {
        return x < 0.0 ? 0.0 : x;
    }

    array3d_t<double> sigmoid_v(array3d_t<double> const &x) {
        array3d_t<double> result(x);
        result.apply(sigmoid);
        return result;
    }

    array3d_t<double> sigmoid_derivative_v(array3d_t<double> const &x) {
        array3d_t<double> result(x);
        result.apply(sigmoid_derivative);
        return result;
    }

    array3d_t<double> stable_softmax_v(array3d_t<double> const &x) {
        array3d_t<double> result(x);
        const int size = (int)result.size();
        double x_max = x.max();

        double sum = 0.0;
        for (int i = 0; i < size; i++) {
            double fi = exp(x(i) - x_max);;
            result(i) = fi;
            sum += fi;
        }

        for (int i = 0; i < size; i++) {
            result(i) /= sum;
        }

        return result;
    }

    array3d_t<double> relu_v(array3d_t<double> const &x) {
        array3d_t<double> result(x);
        result.apply(relu);
        return result;
    }

}
