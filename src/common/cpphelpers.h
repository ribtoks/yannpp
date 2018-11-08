#ifndef CPP_HELPERS_H
#define CPP_HELPERS_H

#include <memory>
#include <iostream>
#include <string>
#include <cstdio>
#include <climits>
#include <vector>

template<typename... Args>
std::string string_format( const std::string& format, Args... args )
{
    const size_t size = std::snprintf( nullptr, 0, format.c_str(), args... ) + 1; // Extra space for '\0'
    std::unique_ptr<char[]> buf( new char[ size ] );
    std::snprintf( buf.get(), size, format.c_str(), args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}

template <typename T>
T swap_endian(T u)
{
    static_assert (CHAR_BIT == 8, "CHAR_BIT != 8");

    union
    {
        T u;
        unsigned char u8[sizeof(T)];
    } source, dest;

    source.u = u;

    for (size_t k = 0; k < sizeof(T); k++)
        dest.u8[k] = source.u8[sizeof(T) - k - 1];

    return dest.u;
}

// function used to generate training input for the neural network
// batches generated with this function are used in update_mini_batch()
std::vector<std::vector<size_t>> batch_indices(size_t size, size_t batch_size);

#endif // CPP_HELPERS_H
