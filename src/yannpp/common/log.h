#ifndef LOG_H
#define LOG_H

namespace yannpp {
    template<typename T>
    class array3d_t;

    void log(const char *fmt, ...);
    void log(array3d_t<float> const &arr);
}

#endif // LOG_H
