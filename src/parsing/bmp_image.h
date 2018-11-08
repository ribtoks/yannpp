#ifndef BMP_HELPER_H
#define BMP_HELPER_H

#include <cstdint>
#include <vector>
#include <string>

class bmp_image_t {
public:
    bmp_image_t(const std::vector<uint8_t> &data, size_t width):
        data_(data),
        width_(width)
    { }

public:
    void save(const std::string &filepath);

private:
    const std::vector<uint8_t> &data_;
    size_t width_;
};


#endif // BMP_HELPER_H
