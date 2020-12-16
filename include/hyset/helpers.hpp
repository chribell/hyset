#pragma once

#ifndef HYSET_HELPERS_HPP
#define HYSET_HELPERS_HPP

#include <climits>
#include <cstddef>
#include <cstdint>

typedef unsigned long word;

#define WORD_BITS (sizeof(word) * CHAR_BIT)
#define BITMAP_NWORDS(_n) (((_n) + WORD_BITS - 1) / WORD_BITS)

namespace hyset {

    template <typename I>
    struct memory_calculator
    {
        long double size;
        char scale;

        memory_calculator(double sizeArg, char scaleArg)
                : size(sizeArg), scale(scaleArg) {}

        inline size_t numberOfElements() {
            if (scale == 'M') {
                size *= 1000000.0;
            } else {
                size *= 1000000000.0;
            }
            return static_cast<size_t>(size / (sizeof (I)));
        }
    };

    __forceinline__ __host__ __device__  int max(int a, int b)
    {
        return (a < b) ? b : a;
    }

    __forceinline__ __host__ __device__  int min(int a, int b)
    {
        return (a > b) ? b : a;
    }

    __forceinline__ __host__ __device__ static bool position_filter(unsigned int probeLength,
            unsigned indexedLength, unsigned int probePosition, unsigned int indexedPosition,
            unsigned int minoverlap) {
        return minoverlap <= hyset::min(probeLength - probePosition, indexedLength - indexedPosition);
    }

    __forceinline__ __device__ void reduce(unsigned int tid, unsigned blockSize, unsigned int* counts, unsigned int* counter)
    {
        __syncthreads();

        if ((blockSize >= 1024) && (tid < 512))
        {
            counts[tid] = *counter = *counter + counts[tid + 512];
        }

        __syncthreads();

        if ((blockSize >= 512) && (tid < 256))
        {
            counts[tid] = *counter = *counter + counts[tid + 256];
        }

        __syncthreads();

        if ((blockSize >= 256) && (tid < 128))
        {
            counts[tid] = *counter = *counter + counts[tid + 128];
        }

        __syncthreads();

        if ((blockSize >= 128) && (tid < 64))
        {
            counts[tid] = *counter = *counter + counts[tid + 64];
        }

        __syncthreads();

        if (tid < 32)
        {
            // Fetch final intermediate sum from 2nd warp
            if (blockSize >= 64) *counter += counts[tid + 32];
            // Reduce final warp using shuffle
            for (int offset = 16; offset > 0; offset /= 2)
            {
                *counter += __shfl_down_sync(0xFFFFFFFF, *counter, offset);
            }
        }
    }

    __forceinline__ __device__ int merge_path(unsigned int* a, unsigned int aSize,
            unsigned int* b, unsigned int bSize, unsigned int diag) {
        int begin = hyset::max(0, diag - bSize);
        int end   = hyset::min(diag, aSize);

        while (begin < end) {
            int mid = (begin + end) / 2;

            if (a[mid] < b[diag - 1 - mid]) begin = mid + 1;
            else end = mid;
        }
        return begin;
    }

    __forceinline__ __device__ unsigned int serial_intersect(unsigned int* a, unsigned aBegin, unsigned aEnd,
            unsigned int* b, unsigned int bBegin, unsigned int bEnd, unsigned int vt) {
        unsigned int count = 0;

        // vt parameter must be odd integer
        for (int i = 0; i < vt; i++)
        {
            bool p = false;
            if ( aBegin >= aEnd ) p = false; // a, out of bounds
            else if ( bBegin >= bEnd ) p = true; //b, out of bounds
            else {
                if (a[aBegin] < b[bBegin]) p = true;
                if (a[aBegin] == b[bBegin]) count++;
            }
            if(p) aBegin++;
            else bBegin++;

        }

        __syncthreads();
        return count;
    }

    __forceinline__ __device__ uint32_t hash(uint32_t v)
    {
        return v * UINT32_C(2654435761);
    }

    __forceinline__ __device__ void change_bit(word* ptr, uint32_t bit, unsigned int words)
    {
        ptr[(words * WORD_BITS - bit - 1) / WORD_BITS] ^= 1UL << ((bit) % WORD_BITS);
    }

    __forceinline__ __device__  unsigned int hamming_distance(unsigned long* first, unsigned long* second, unsigned int words)
    {
        unsigned int count = 0;
        for (unsigned i = 0; i < words; ++i) {
            count += __popcll( *(first + i) ^ *(second + i) );
        }
        return count;
    }
};



#endif // HYSET_HELPERS_HPP