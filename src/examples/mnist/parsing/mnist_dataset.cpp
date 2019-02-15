#include "mnist_dataset.h"

#include <yannpp/common/array3d.h>
#include <yannpp/common/cpphelpers.h>
#include <yannpp/common/log.h>

#include "parsing/parsed_images.h"
#include "parsing/parsed_labels.h"

#ifdef WITH_BITMAPS
#include "parsing/bmp_image.h"
#endif

#ifdef _WIN32
    #define TRAIN_IMAGES_FILE "train-images.idx3-ubyte"
    #define TRAIN_LABELS_FILE "train-labels.idx1-ubyte"
#else
    #define TRAIN_IMAGES_FILE "train-images-idx3-ubyte"
    #define TRAIN_LABELS_FILE "train-labels-idx1-ubyte"
#endif

namespace yannpp {
    mnist_dataset_t::mnist_dataset_t(const std::string &data_root):
        data_root_(data_root)
    {
        log("Parsing mnist dataset from directory %s", data_root.c_str());
    }

    std::vector<std::tuple<array3d_t<float>, array3d_t<float> > > mnist_dataset_t::training_data(int limit) {
        parsed_images_t parsed_images(data_root_ + TRAIN_IMAGES_FILE);
        auto itImg = parsed_images.begin();
        auto itImgEnd = parsed_images.end();

        parsed_labels_t parsed_labels(data_root_ + TRAIN_LABELS_FILE);
        auto itLbl = parsed_labels.begin();
        auto itLblEnd = parsed_labels.end();

        std::vector<std::tuple<array3d_t<float>, array3d_t<float>>> data;
        const size_t count_limit = limit == -1 ? parsed_images.size() : limit;
        data.reserve(count_limit);

        for (;
             itImg != itImgEnd && itLbl != itLblEnd;
             ++itImg, ++itLbl) {
            array3d_t<float> input(*itImg); input.mul(1.f / 255.f); input.reshape(shape3d_t(28, 28, 1));
            array3d_t<float> result(shape_row(10), 0.0); result(*itLbl) = 1.f;

            data.emplace_back(std::make_tuple(std::move(input), std::move(result)));

            if (data.size() >= count_limit) { break; }
        }

        log("Training data loaded: %d images", data.size());

        return data;
    }

#ifdef WITH_BITMAPS
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
#endif
}
