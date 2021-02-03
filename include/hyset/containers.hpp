#pragma once

#ifndef HYSET_CONTAINERS_HPP
#define HYSET_CONTAINERS_HPP

#include <vector>
#include <memory>


namespace hyset {

namespace containers {

    template <typename T>
    struct device_array {
        cuda::memory::managed::unique_ptr<T[]> ptr;
        size_t length;

        device_array(size_t len) : ptr(cuda::memory::managed::make_unique<T[]>(len)), length(len) {
            cuda::memory::zero({ptr.get(), sizeof(T) * length});
        }
        device_array(std::vector<T>& hostArray) {
            length = hostArray.size();
            ptr = cuda::memory::managed::make_unique<T[]>(length);
            cuda::memory::copy(ptr.get(), &hostArray[0], sizeof(T) * length);
        }

        __forceinline__ __device__ T& operator[] (unsigned int i)
        {
            return ptr[i];
        }

        __forceinline__ __device__ T& at(unsigned int i)
        {
            return ptr[i];
        }
    };

    template <typename T>
    struct host_array {
        std::unique_ptr<T[]> ptr;
        size_t length;

        host_array(size_t len) : ptr(std::unique_ptr<T[]>(new T[len]())), length(len) {}
        host_array(hyset::containers::device_array<T>& deviceArray) {
            length = deviceArray.length;
            ptr = std::unique_ptr<T[]>(new T[length]);
            cuda::memory::copy(ptr.get(), deviceArray.ptr.get(), sizeof(T) * length);
        }

        inline T& operator[] (unsigned int i)
        {
            return ptr[i];
        }

        inline T& at(unsigned int i)
        {
            return ptr[i];
        }
    };

    struct candidate_set {
        std::vector<unsigned int> candidates;
        typedef std::vector<unsigned int>::iterator iterator;

        inline void addCandidate(unsigned int candidateID) {
            candidates.push_back(candidateID);
        }

        inline size_t size() const {
            return candidates.size();
        }

        inline iterator begin() {
            return candidates.begin();
        }

        inline iterator end() {
            return candidates.end();
        }

        inline void clear() {
            candidates.clear();
        }
    };

    struct host_candidates {
        std::vector<unsigned int> probes;
        std::vector<unsigned int> offsets;
        std::vector<unsigned int> candidates;

        host_candidates(unsigned int maxProbes, unsigned int maxCandidates) {
            probes.reserve(maxProbes);
            offsets.reserve(maxProbes);
            candidates.reserve(maxCandidates);
        }

        inline void clear() {
            probes.clear();
            offsets.clear();
            candidates.clear();
        }

    };

    struct device_candidates {
        hyset::containers::device_array<unsigned int>* probes;
        hyset::containers::device_array<unsigned int>* offsets;
        hyset::containers::device_array<unsigned int>* candidates;

        device_candidates(unsigned int maxProbes, unsigned int maxCandidates) {
            probes = new hyset::containers::device_array<unsigned int>(maxProbes);
            offsets = new hyset::containers::device_array<unsigned int>(maxProbes);
            candidates = new hyset::containers::device_array<unsigned int>(maxCandidates);
        }

        inline void copy(host_candidates& hostCandidates) {
            cuda::memory::copy(probes->ptr.get(), &hostCandidates.probes[0], sizeof(unsigned int) * hostCandidates.probes.size());
            cuda::memory::copy(offsets->ptr.get(), &hostCandidates.offsets[0], sizeof(unsigned int) * hostCandidates.offsets.size());
            cuda::memory::copy(candidates->ptr.get(), &hostCandidates.candidates[0], sizeof(unsigned int) * hostCandidates.candidates.size());
        }

        ~device_candidates() {
            probes->hyset::containers::device_array<unsigned int>::~device_array();
            offsets->hyset::containers::device_array<unsigned int>::~device_array();
            candidates->hyset::containers::device_array<unsigned int>::~device_array();
        }

    };

    struct device_candidates_wrapper {
        unsigned int* probes;
        size_t probesLen;
        unsigned int* offsets;
        size_t offsetsLen;
        unsigned int* candidates;
        size_t candidatesLen;

        device_candidates_wrapper(device_candidates& deviceCandidates) :
            probes(deviceCandidates.probes->ptr.get()),
            probesLen(deviceCandidates.probes->length),
            offsets(deviceCandidates.offsets->ptr.get()),
            offsetsLen(deviceCandidates.offsets->length),
            candidates(deviceCandidates.candidates->ptr.get()),
            candidatesLen(deviceCandidates.candidates->length) {}
    };

    template<typename output_type>
    struct device_result {
        hyset::containers::device_array<output_type>* output;

        device_result(unsigned int size) {
            output = new hyset::containers::device_array<output_type>(size);
        }

        ~device_result() {
            output->hyset::containers::template device_array<output_type>::~device_array();
        }
    };

    struct device_result_wrapper {
        unsigned int* counts;
        size_t countsLen;
        hyset::structs::pair* pairs;
        size_t pairsLen;

        device_result_wrapper(device_result<unsigned int>& deviceResult) :
                counts(deviceResult.output->ptr.get()),
                countsLen(deviceResult.output->length) {}

        device_result_wrapper(device_result<hyset::structs::pair>& deviceResult) :
                pairs(deviceResult.output->ptr.get()),
                pairsLen(deviceResult.output->length) {}

    };



    struct device_counts {

        hyset::containers::device_array<unsigned int>* output;

        device_counts(unsigned int size) {
            output = new hyset::containers::device_array<unsigned int>(size);
        }

        ~device_counts() {
            output->hyset::containers::device_array<unsigned int>::~device_array();
        }
    };

    struct device_pairs {
        hyset::containers::device_array<hyset::structs::pair>* output;

        device_pairs(unsigned int size) {
            output = new hyset::containers::device_array<hyset::structs::pair>(size);
        }

        ~device_pairs() {
            output->hyset::containers::device_array<hyset::structs::pair>::~device_array();
        }
    };

    struct device_bitmaps {
        hyset::containers::device_array<word>* bitmaps;
        unsigned int bitmapWords;
        unsigned int bitmapSize;

        device_bitmaps(size_t length, unsigned int words) {
            bitmapWords = words;
            bitmapSize = bitmapWords * WORD_BITS;
            bitmaps = new hyset::containers::device_array<word>(length * bitmapWords);
        }

        ~device_bitmaps() {
            bitmaps->hyset::containers::device_array<word>::~device_array();
        }
    };

    struct device_bitmap_wrapper {
        word* ptr;
        size_t length;
        unsigned int bitmapWords;
        unsigned int bitmapSize;

        device_bitmap_wrapper(device_bitmaps& deviceBitmaps) :
                ptr(deviceBitmaps.bitmaps->ptr.get()),
                length(deviceBitmaps.bitmaps->length),
                bitmapWords(deviceBitmaps.bitmapWords),
                bitmapSize(deviceBitmaps.bitmapSize) {}

        __forceinline__ __host__ __device__ word* at(size_t pos) const
        {
            return ptr + (pos * bitmapWords);
        }

    };

}
}


#endif // HYSET_CONTAINERS_HPP