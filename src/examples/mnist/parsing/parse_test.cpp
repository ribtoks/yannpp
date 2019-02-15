#include "parsed_images.h"
#include "parsed_labels.h"
#include "bmp_image.h"
#include "cpphelpers.h"
#include <exception>

int main(int argc, char* argv[]) {
    if (argc != 3) {
        throw std::runtime_error("Missing path to training files");
    }

    std::string images_filepath(argv[1]);
    parsed_images_t parsed_images(images_filepath);
    auto itImg = parsed_images.begin();
    auto itImgEnd = parsed_images.end();

    std::string labels_filepath(argv[2]);
    parsed_labels_t parsed_labels(labels_filepath);
    auto itLbl = parsed_labels.begin();
    auto itLblEnd = parsed_labels.end();

    auto width = parsed_images.img_width();
    
    int count = 0;
    for (;
         itImg != itImgEnd && itLbl != itLblEnd;
         ++itImg, ++itLbl) {
        bmp_image_t(*itImg, width)
            .save(
                string_format("test_%d_digit_%d.bmp", count, *itLbl));

        // debug
        if (++count > 20) { break; }
    }
    
    return 0;
}
