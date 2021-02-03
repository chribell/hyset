#pragma once

#ifndef HYSET_INDEX_HPP
#define HYSET_INDEX_HPP

#include <hyset/containers.hpp>
#include <hyset/structs.hpp>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>


namespace hyset {

namespace index {
    struct device_index {
        hyset::containers::device_array<unsigned int>* offsets; // Inverted lists' offsets.
        hyset::containers::device_array<unsigned int>* lengths; // Length of an inverted list
        hyset::containers::device_array<hyset::structs::entry>* index; // the complete index in linearized format (concatenated inverted lists)
        hyset::containers::device_array<hyset::structs::entry>* entries; // initial prefix entries, used for index construction
        unsigned int universeSize;

        device_index(unsigned int universe, unsigned int length) {
            universeSize = universe;
            offsets = new hyset::containers::device_array<unsigned int>(universeSize);
            lengths = new hyset::containers::device_array<unsigned int>(universeSize);

            index = new hyset::containers::device_array<hyset::structs::entry>(length);
            entries = new hyset::containers::device_array<hyset::structs::entry>(length);
        }

        inline void zero () {
            cuda::memory::zero({offsets->ptr.get(), sizeof(unsigned int) * offsets->length});
            cuda::memory::zero({lengths->ptr.get(), sizeof(unsigned int) * lengths->length});
            cuda::memory::zero({index->ptr.get(), sizeof(hyset::structs::entry) * index->length});
        }

        ~device_index() { // api-wrappers RAII guarantees that memory will be freed
            offsets->hyset::containers::device_array<unsigned int>::~device_array();
            lengths->hyset::containers::device_array<unsigned int>::~device_array();
            index->hyset::containers::device_array<hyset::structs::entry>::~device_array();
            entries->hyset::containers::device_array<hyset::structs::entry>::~device_array();
        }
    };


    struct host_array_index {
        hyset::containers::host_array<unsigned int>* offsets;
        hyset::containers::host_array<hyset::structs::entry>* index;

        host_array_index(device_index& deviceIndex) {
            offsets = new hyset::containers::host_array<unsigned int>(*deviceIndex.offsets);
            index = new hyset::containers::host_array<hyset::structs::entry>(*deviceIndex.index);
        }

        host_array_index(unsigned int universeSize, unsigned int entries) {
            offsets = new hyset::containers::host_array<unsigned int>(universeSize);
            index = new hyset::containers::host_array<hyset::structs::entry>(entries);
        }

        ~host_array_index() {
            offsets->hyset::containers::host_array<unsigned int>::~host_array();
            index->hyset::containers::host_array<hyset::structs::entry>::~host_array();
        }

        struct iterator {
            unsigned int current;
            unsigned int count;

            hyset::structs::entry* entries;
            unsigned int firstToCheck;

            inline iterator() : current(0), count(0) {}

            inline iterator(hyset::structs::entry* entries, unsigned int firstToCheck, unsigned int count) :
                    entries(entries), firstToCheck(firstToCheck), current(firstToCheck), count(count) {}

            inline bool end() const {
                return current == count;
            }

            inline void next() {
                current++;
            }

            inline hyset::structs::entry& operator*(){
                return entries[current];
            }

            inline hyset::structs::entry* operator->() {
                return &entries[current];
            }

            inline bool length_filter(unsigned int indexedSize, unsigned int minsize) {
                if (indexedSize < minsize) {
                    current++;
                    firstToCheck = current;
                    return false;
                }
                return true;
            }
        };

        inline iterator get_iterator(unsigned int token, unsigned int firstToCheck) {
            if (list_size(token) == 0) {
                return iterator();
            } else {
                return iterator(index->ptr.get() + list_start(token), firstToCheck, list_size(token));
            }
        }

        inline unsigned int list_start(unsigned int token) const
        {
            return offsets->at(token - 1);
        }

        inline unsigned int list_end(unsigned int token) const
        {
            return offsets->at(token);
        }

        inline unsigned int list_size(unsigned int token) const
        {
            return list_end(token) - list_start(token);
        }
    };

    struct host_vector_index {
        typedef std::vector<hyset::structs::entry> inverted_list;

        std::vector<inverted_list>* index;

        host_vector_index(unsigned int size) {
            index = new std::vector<inverted_list>(size);
        }

        inverted_list& get_list(unsigned int token) {
            return (*index)[token];
        }

        struct iterator {
            unsigned int current;
            unsigned int count;

            inverted_list list;

            inline iterator() : current(0), count(0) {}

            inline iterator(inverted_list& list) :
                    list(list),  current(0) {
                count = list.size();
            }

            inline bool end() const {
                return current == count;
            }

            inline void next() {
                current++;
            }

            inline hyset::structs::entry& operator*(){
                return list[current];
            }

            inline hyset::structs::entry* operator->() {
                return &list[current];
            }
        };

        inline iterator get_iterator(unsigned int token) {
            inverted_list list = get_list(token);
            if (list.empty()) {
                return iterator();
            } else {
                return iterator(list);
            }
        }

    };

    struct device_index_wrapper { // used as proxy for kernel invocation
        unsigned int* offsets;
        size_t offsetsLen;
        unsigned int* lengths;
        size_t lengthsLen;
        hyset::structs::entry* index;
        size_t indexLen;
        hyset::structs::entry* entries;
        size_t entriesLen;


        device_index_wrapper(device_index& deviceIndex) :
            offsets(deviceIndex.offsets->ptr.get()),
            offsetsLen(deviceIndex.offsets->length),
            lengths(deviceIndex.lengths->ptr.get()),
            lengthsLen(deviceIndex.lengths->length),
            index(deviceIndex.index->ptr.get()),
            indexLen(deviceIndex.index->length),
            entries(deviceIndex.entries->ptr.get()),
            entriesLen(deviceIndex.entries->length) {}

        __forceinline__ __device__ unsigned int list_start(unsigned int token) const {
            return offsets[token - 1];
        }

        __forceinline__ __device__ unsigned int list_end(unsigned int token) const {
            return offsets[token];
        }

        __forceinline__ __device__ unsigned int list_size(unsigned int token) const {
            return list_end(token) - list_start(token);
        }

    };


    __global__ void histogram(hyset::structs::entry* entries, unsigned int* lengths, unsigned int n) {
        unsigned int blockSize = n / gridDim.x + (n % gridDim.x == 0 ? 0 : 1);		//Number of items for each block
        unsigned int offset = blockSize * (blockIdx.x); 				//Beginning of the block
        unsigned int lim = offset + blockSize; 						//End of block the
        if (lim >= n) lim = n;
        if (offset >= lim) return; // if offset is larger or equal to the limit then block is redundant
        unsigned int size = lim - offset;						//Block size

        entries += offset;

        for (unsigned int i = threadIdx.x; i < size; i += blockDim.x) {
            unsigned int token = entries[i].token;
            atomicAdd(&lengths[token], 1);
        }
    }

    __global__ void create_index(hyset::index::device_index_wrapper invertedIndex, unsigned int n) {
        unsigned int blockSize = n / gridDim.x + (n % gridDim.x == 0 ? 0 : 1);		//Number of items used by each block
        unsigned int offset = blockSize * (blockIdx.x); 				//Beginning of the block
        unsigned int lim = offset + blockSize; 						//End of the block
        if (lim >= n) lim = n;
        if (offset >= lim) return; // if offset is larger or equal to the limit then block is redundant
        unsigned int size = lim - offset;						//Block size

        invertedIndex.entries += offset;

        for (unsigned int i = threadIdx.x; i < size; i += blockDim.x) {
            hyset::structs::entry entry = invertedIndex.entries[i];
            unsigned int pos = atomicAdd(&invertedIndex.offsets[entry.token], 1);
            invertedIndex.index[pos] = entry;
        }
    }

    template <typename Segment>
    host_array_index* make_inverted_index(Segment& segment, hyset::collection::host_collection& hostCollection) {

        auto current_device = cuda::device::current::get();
        int gridX = current_device.get_attribute(cudaDevAttrMultiProcessorCount) * 16;
        int threadsX = current_device.get_attribute(cudaDevAttrMaxThreadsPerBlock) / 2;

        unsigned int start = segment.first_entry_position();
        unsigned int end = segment.last_entry_position();
        unsigned int length = end - start;

        hyset::index::device_index deviceIndex(hostCollection.universeSize, length);
        cuda::memory::copy(deviceIndex.entries->ptr.get(), &hostCollection.prefixes[start], sizeof(hyset::structs::entry) * length);

        cuda::launch(hyset::index::histogram,
                     cuda::launch_configuration_t(gridX, threadsX),
                     deviceIndex.entries->ptr.get(), deviceIndex.lengths->ptr.get(), length
        );

        thrust::device_ptr<unsigned int> thrustLengths(deviceIndex.lengths->ptr.get());
        thrust::device_ptr<unsigned int> thrustOffsets(deviceIndex.offsets->ptr.get());
        thrust::exclusive_scan(thrustLengths, thrustLengths + hostCollection.universeSize, thrustOffsets);


        cuda::launch(hyset::index::create_index,
                     cuda::launch_configuration_t(gridX, threadsX), deviceIndex, length
        );

        thrust::device_ptr<hyset::structs::entry> thrustIndex(deviceIndex.index->ptr.get());
        thrust::sort(thrustIndex, thrustIndex + length, hyset::structs::entry_comparator());

        return new host_array_index(deviceIndex);
    }

    void make_inverted_index(hyset::index::device_index& deviceIndex, hyset::structs::block& block, std::shared_ptr<hyset::collection::host_collection>& hostCollection) {
        auto current_device = cuda::device::current::get();
        int gridX = current_device.get_attribute(cudaDevAttrMultiProcessorCount) * 16;
        int threadsX = current_device.get_attribute(cudaDevAttrMaxThreadsPerBlock) / 2;

        unsigned int start = block.firstEntryPosition;
        unsigned int end = block.lastEntryPosition;
        unsigned int length = end - start;

        deviceIndex.zero();
        cuda::memory::copy(deviceIndex.entries->ptr.get(), &hostCollection->prefixes[start], sizeof(hyset::structs::entry) * length);

        cuda::launch(hyset::index::histogram,
                     cuda::launch_configuration_t(gridX, threadsX),
                     deviceIndex.entries->ptr.get(), deviceIndex.lengths->ptr.get(), length
        );

        thrust::device_ptr<unsigned int> thrustLengths(deviceIndex.lengths->ptr.get());
        thrust::device_ptr<unsigned int> thrustOffsets(deviceIndex.offsets->ptr.get());
        thrust::exclusive_scan(thrustLengths, thrustLengths + hostCollection->universeSize, thrustOffsets);

        cuda::launch(hyset::index::create_index,
                     cuda::launch_configuration_t(gridX, threadsX), deviceIndex, length
        );

        // TODO: remove if redundant
//        thrust::device_ptr<hyset::structs::entry> thrustIndex(deviceIndex.index->ptr.get());
//        thrust::sort(thrustIndex, thrustIndex + length, hyset::structs::entry_comparator());

    }


};

};


#endif // HYSET_INDEX_HPP