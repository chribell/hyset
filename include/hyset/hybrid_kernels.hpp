#pragma once

#ifndef HYSET_HYBRID_KERNELS_HPP
#define HYSET_HYBRID_KERNELS_HPP

#include <hyset/collection.hpp>
#include <hyset/structs.hpp>
#include <hyset/verification.hpp>

namespace hyset {
    namespace algorithms {
        namespace hybrid {
            namespace kernels {
                template <typename Similarity, bool aggregate>
                __global__ void scenario_1(hyset::collection::device_collection_wrapper indexedCollection,
                                           hyset::collection::device_collection_wrapper probeCollection,
                                           hyset::containers::device_candidates_wrapper candidates,
                                           hyset::containers::device_result_wrapper result,
                                           unsigned int* globalCount,
                                           unsigned int maxSetSize,
                                           double threshold) {
                    extern __shared__ unsigned int shared[];
                    unsigned int* probe = shared;
                    unsigned int* counts = (unsigned int*) &probe[maxSetSize];

                    unsigned int probeID = candidates.probes[blockIdx.x];

                    unsigned int probeStart = probeCollection.starts[probeID];
                    unsigned int probeSize = probeCollection.sizes[probeID];

                    bool inShared = probeSize <= maxSetSize;

                    if (inShared) { // if probe set fits in shared memory, load it
                        for (unsigned int i = threadIdx.x; i < probeSize; i+= blockDim.x) {
                            probe[i] = probeCollection.tokens[probeStart + i];
                        }
                        __syncthreads(); // since we handle one probe set per block, it's safe to call sync threads here
                    }

                    unsigned int candidatesStart = blockIdx.x == 0 ? 0 : candidates.offsets[blockIdx.x - 1];
                    unsigned int candidatesEnd = candidates.offsets[blockIdx.x];

                    unsigned int counter = 0;

                    for (unsigned int offset = candidatesStart + threadIdx.x; offset < candidatesEnd; offset += blockDim.x) {
                        unsigned int candidateID = candidates.candidates[offset];
                        unsigned int candidateStart = indexedCollection.starts[candidateID];
                        unsigned int candidateSize = indexedCollection.sizes[candidateID];

                        unsigned int minoverlap = Similarity::minoverlap(probeSize, candidateSize, threshold);

                        if (hyset::verification::device::verify_pair(
                                inShared ? probe : &probeCollection.tokens[probeStart],
                                probeSize,
                                &indexedCollection.tokens[candidateStart],
                                candidateSize,
                                minoverlap)) {
                            if (aggregate) {
                                counter++;
                            } else {
                                unsigned int pos = atomicAdd(globalCount, 1);
                                result.pairs[pos] = hyset::structs::pair(probeID, candidateID);
                            }
                        }
                    }

                    if (aggregate) {
                        counts[threadIdx.x] = counter;
                        hyset::reduce(threadIdx.x, blockDim.x, counts, &counter);
                        if (threadIdx.x == 0) {
                            result.counts[blockIdx.x] = counter;
                        }
                    }
                }

                template <typename Similarity, bool aggregate>
                __global__ void scenario_2(hyset::collection::device_collection_wrapper indexedCollection,
                                           hyset::collection::device_collection_wrapper probeCollection,
                                           hyset::containers::device_candidates_wrapper candidates,
                                           hyset::containers::device_result_wrapper result,
                                           unsigned int* globalCount,
                                           unsigned int maxSetSize,
                                           double threshold) {
                    extern __shared__ unsigned int shared[];
                    unsigned int* probe = shared;
                    unsigned int* candidate = (unsigned int*) &probe[maxSetSize];
                    unsigned int* intersections = (unsigned int*) &candidate[maxSetSize];

                    unsigned int probeID = candidates.probes[blockIdx.x];

                    unsigned int probeStart = probeCollection.starts[probeID];
                    unsigned int probeSize = probeCollection.sizes[probeID];

                    bool probeInShared = probeSize <= maxSetSize;

                    if (probeInShared) { // if probe set fits in shared memory, load it
                        for (unsigned int i = threadIdx.x; i < probeSize; i+= blockDim.x) {
                            probe[i] = probeCollection.tokens[probeStart + i];
                        }
                        __syncthreads(); // since we handle one probe set per block, it's safe to call sync threads here
                    }

                    intersections[threadIdx.x] = 0;

                    unsigned int candidatesStart = blockIdx.x == 0 ? 0 : candidates.offsets[blockIdx.x - 1];
                    unsigned int candidatesEnd = candidates.offsets[blockIdx.x];

                    unsigned int counter = 0;

                    for (unsigned int offset = candidatesStart; offset < candidatesEnd; ++offset) {
                        unsigned int candidateID = candidates.candidates[offset];
                        unsigned int candidateStart = indexedCollection.starts[candidateID];
                        unsigned int candidateSize = indexedCollection.sizes[candidateID];

                        bool candidateInShared = candidateSize <= maxSetSize;
                        if (candidateInShared) {
                            for (unsigned int i = threadIdx.x; i < candidateSize; i+= blockDim.x) {
                                candidate[i] = indexedCollection.tokens[candidateStart + i];
                            }
                            __syncthreads();
                        }

                        unsigned int vt = ((probeSize + candidateSize + 1) / blockDim.x) + 1;
                        unsigned int diagonal = threadIdx.x * vt;

                        int mp = hyset::merge_path(
                                probeInShared ? probe : &probeCollection.tokens[probeStart],
                                probeSize,
                                &indexedCollection.tokens[candidateStart],
                                candidateSize,
                                diagonal);

                        unsigned intersection = hyset::serial_intersect(
                                probeInShared ? probe : &probeCollection.tokens[probeStart],
                                mp,
                                probeSize,
                                &indexedCollection.tokens[candidateStart],
                                diagonal - mp,
                                candidateSize,
                                vt);

                        reduce(threadIdx.x, blockDim.x, intersections, &intersection);

                        __syncthreads();

                        if (threadIdx.x == 0) {
                            unsigned int minoverlap = Similarity::minoverlap(probeSize, candidateSize, threshold);

                            if (intersection >= minoverlap) {

                                if (aggregate) {
                                    counter++;
                                } else {
                                    unsigned int pos = atomicAdd(globalCount, 1);
                                    result.pairs[pos] = hyset::structs::pair(probeID, candidateID);
                                }
                            }
                        }
                    }

                    if (aggregate && threadIdx.x == 0) {
                        result.counts[blockIdx.x] = counter;
                    }

                }
            }
        }
    }
}

#endif // HYSET_HYBRID_KERNELS_HPP