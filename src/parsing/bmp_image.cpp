#include "bmp_image.h"
#include <fstream>
#include <cstring>
#include <exception>

void bmp_image_t::save(const std::string &filepath) {
    const int h = data_.size()/width_;
    const int w = width_;
    const int filesize = 54 + 3*w*h;

    std::vector<uint8_t> img;
    img.resize(3*w*h);

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            int x = j;
            int y = i;
            int pos = x + y*w;
            int v = 255 - data_[pos];
            int r = v, g = v, b = v;
            img[pos*3+2] = (unsigned char)(r);
            img[pos*3+1] = (unsigned char)(g);
            img[pos*3+0] = (unsigned char)(b);
        }
    }

    unsigned char bmpfileheader[14] = {'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0};
    unsigned char bmpinfoheader[40] = {40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0};
    unsigned char bmppad[3] = {0,0,0};

    bmpfileheader[ 2] = (unsigned char)(filesize    );
    bmpfileheader[ 3] = (unsigned char)(filesize>> 8);
    bmpfileheader[ 4] = (unsigned char)(filesize>>16);
    bmpfileheader[ 5] = (unsigned char)(filesize>>24);

    bmpinfoheader[ 4] = (unsigned char)(       w    );
    bmpinfoheader[ 5] = (unsigned char)(       w>> 8);
    bmpinfoheader[ 6] = (unsigned char)(       w>>16);
    bmpinfoheader[ 7] = (unsigned char)(       w>>24);
    bmpinfoheader[ 8] = (unsigned char)(       h    );
    bmpinfoheader[ 9] = (unsigned char)(       h>> 8);
    bmpinfoheader[10] = (unsigned char)(       h>>16);
    bmpinfoheader[11] = (unsigned char)(       h>>24);

    FILE *f = fopen(filepath.c_str(), "wb");
    if (f == NULL) { throw std::runtime_error("Failed to create output file"); }
    
    fwrite(bmpfileheader,1,14,f);
    fwrite(bmpinfoheader,1,40,f);

    uint8_t *start = &img[0];
    for (int i=0; i<h; i++) {
        fwrite(start+(w*(h-i-1)*3),3,w,f);
        fwrite(bmppad,1,(4-(w*3)%4)%4,f);
    }

    fclose(f);
}









