#include "mnist_dataset.h"
#include "common/array3d.h"
#include "common/cpphelpers.h"
#include "common/log.h"
#include "parsing/parsed_images.h"
#include "parsing/parsed_labels.h"
#include "parsing/bmp_image.h"

namespace yannpp {
    mnist_dataset_t::mnist_dataset_t(const std::string &data_root):
        data_root_(data_root)
    {
    }

    std::vector<std::tuple<array3d_t<double>, array3d_t<double>>> mnist_dataset_t::training_data(int limit) {
        parsed_images_t parsed_images(data_root_ + "train-images-idx3-ubyte");
        auto itImg = parsed_images.begin();
        auto itImgEnd = parsed_images.end();

        parsed_labels_t parsed_labels(data_root_ + "train-labels-idx1-ubyte");
        auto itLbl = parsed_labels.begin();
        auto itLblEnd = parsed_labels.end();

        std::vector<std::tuple<array3d_t<double>, array3d_t<double>>> data;
        const size_t count_limit = limit == -1 ? parsed_images.size() : limit;
        data.reserve(count_limit);

        for (;
             itImg != itImgEnd && itLbl != itLblEnd;
             ++itImg, ++itLbl) {
            array3d_t<double> input(*itImg); input.mul(1.0 / 255.0); input.reshape(shape_matrix(28, 28));
            array3d_t<double> result(shape_row(10), 0.0); result(*itLbl) = 1.0;

            data.emplace_back(std::make_tuple(std::move(input), std::move(result)));

            if (data.size() >= count_limit) { break; }
        }

        log("Training data loaded: %d images", data.size());

        return data;
    }

    void mnist_dataset_t::save_as_images(int limit) {
        parsed_images_t parsed_images(data_root_ + "train-images-idx3-ubyte");
        auto itImg = parsed_images.begin();
        auto itImgEnd = parsed_images.end();

        parsed_labels_t parsed_labels(data_root_ + "train-labels-idx1-ubyte");
        auto itLbl = parsed_labels.begin();
        auto itLblEnd = parsed_labels.end();

        const size_t count_limit = limit == -1 ? parsed_images.size() : limit;
        size_t i = 0;

        for (;
             itImg != itImgEnd && itLbl != itLblEnd;
             ++itImg, ++itLbl) {

            bmp_image_t(*itImg, 28)
                    .save(
                        string_format("test_%d_digit_%d.bmp", i++, *itLbl));

            if (i > count_limit) { break; }
        }
    }
}
