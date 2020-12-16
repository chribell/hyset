#pragma once

#ifndef HYSET_VERIFICATION_HPP
#define HYSET_VERIFICATION_HPP

namespace hyset{
  namespace verification {
      namespace host {
          template<typename T>
          bool inline verify_pair(const T &r1, const T &r2,
                                  unsigned int overlapthres, unsigned int posr1 = 0, unsigned int posr2 = 0,
                                  unsigned int foundoverlap = 0) {
              unsigned int maxr1 = r1.size() - posr1 + foundoverlap;
              unsigned int maxr2 = r2.size() - posr2 + foundoverlap;

              while (maxr1 >= overlapthres && maxr2 >= overlapthres && foundoverlap < overlapthres) {
                  if (r1[posr1] == r2[posr2]) {
                      ++posr1;
                      ++posr2;
                      ++foundoverlap;
                  } else if (r1[posr1] < r2[posr2]) {
                      ++posr1;
                      --maxr1;
                  } else {
                      ++posr2;
                      --maxr2;
                  }
              }

              return foundoverlap >= overlapthres;
          }
      }

      namespace device {
          __forceinline__ __device__ bool verify_pair(unsigned int* r1, unsigned int r1Size,
                                                      unsigned int* r2, unsigned int r2Size,
                                                      unsigned int overlapthres, unsigned int posr1 = 0,
                                                      unsigned int posr2 = 0, unsigned int foundoverlap = 0) {

              unsigned int maxr1 = r1Size - posr1 + foundoverlap;
              unsigned int maxr2 = r2Size - posr2 + foundoverlap;

              while (maxr1 >= overlapthres && maxr2 >= overlapthres && foundoverlap < overlapthres) {
                  if (r1[posr1] == r2[posr2]) {
                      ++posr1;
                      ++posr2;
                      ++foundoverlap;
                  } else if (r1[posr1] < r2[posr2]) {
                      ++posr1;
                      --maxr1;
                  } else {
                      ++posr2;
                      --maxr2;
                  }
              }

              return foundoverlap >= overlapthres;
          }

          template<typename Similarity, bool aggregate, bool useCount>
          __global__ void verify_pairs(hyset::collection::device_collection_wrapper indexedCollection,
                                       hyset::collection::device_collection_wrapper probeCollection,
                                       hyset::containers::device_result_wrapper result,
                                       hyset::structs::block indexedBlock,
                                       hyset::structs::block probeBlock,
                                       unsigned int* filter,
                                       unsigned int blockSize,
                                       unsigned int *count,
                                       double threshold) {
              extern __shared__ unsigned int shared[];
              unsigned int* sharedCount = shared;

              if (aggregate) {
                  *shared = 0;
              }

              unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

              for (; i < blockSize * blockSize; i += gridDim.x * blockDim.x) {

                  if (filter[i]) {
                      unsigned int probeID = i / blockSize + probeBlock.startID;
                      unsigned int candidateID = i % blockSize + indexedBlock.startID;

                      unsigned int *probe = probeCollection.tokens + probeCollection.starts[probeID];
                      unsigned int *candidate = indexedCollection.tokens + indexedCollection.starts[candidateID];

                      unsigned int probeSize = probeCollection.sizes[probeID];
                      unsigned int candidateSize = indexedCollection.sizes[candidateID];

                      unsigned int minoverlap = Similarity::minoverlap(probeSize, candidateSize, threshold);
                      unsigned int overlap = useCount ? filter[i] : 0, recpos = 0, indrecpos = 0;

                      if (useCount) { // use existing calculated prefix overlap
                          //First position after last position by index lookup in indexed record
                          unsigned int lastposind = Similarity::midprefix(candidateSize, threshold);

                          //First position after last position by index lookup in probing record
                          unsigned int lastposprobe = Similarity::maxprefix(probeSize, threshold);

                          unsigned int recpreftoklast = probe[lastposprobe - 1];
                          unsigned int indrecpreftoklast = candidate[lastposind - 1];


                          if (recpreftoklast > indrecpreftoklast) {
                              recpos = overlap;
                              //first position after minprefix / lastposind
                              indrecpos = lastposind;
                          } else {
                              // First position after maxprefix / lastposprobe
                              recpos = lastposprobe;
                              indrecpos = overlap;
                          }
                      }

                      if (hyset::verification::device::verify_pair(
                              probe,
                              probeSize,
                              candidate,
                              candidateSize,
                              minoverlap,
                              recpos,
                              indrecpos,
                              overlap)) {
                          if (aggregate) {
                              atomicAdd(sharedCount, 1);
                          } else {
                              unsigned int pos = atomicAdd(count, 1);
                              result.pairs[pos] = hyset::structs::pair(probeID, candidateID);
                          }
                      }
                  }
              }
              __syncthreads();
              if (aggregate && threadIdx.x == 0) {
                  result.counts[blockIdx.x] = *sharedCount;
              }
          }

          template<typename Similarity, bool aggregate, bool useCount>
          __global__ void verify_pairs_1(hyset::collection::device_collection_wrapper indexedCollection,
                                       hyset::collection::device_collection_wrapper probeCollection,
                                       hyset::containers::device_result_wrapper result,
                                       hyset::structs::block indexedBlock,
                                       hyset::structs::block probeBlock,
                                       unsigned int* filter,
                                       unsigned int blockSize,
                                       unsigned int *count,
                                       double threshold) {
              unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

              for (; i < blockSize * blockSize; i += gridDim.x * blockDim.x) {

                  if (filter[i]) {
                      atomicAdd(&result.counts[0], 1);
                  }
              }
          }


          template<typename Similarity, bool aggregate>
          __global__ void calculate_similarity(hyset::collection::device_collection_wrapper indexedCollection,
                                       hyset::collection::device_collection_wrapper probeCollection,
                                       hyset::containers::device_result_wrapper result,
                                       hyset::structs::block indexedBlock,
                                       hyset::structs::block probeBlock,
                                       unsigned int* intersection,
                                       unsigned int blockSize,
                                       unsigned int *count,
                                       double threshold) {
              extern __shared__ unsigned int shared[];
              unsigned int* sharedCount = shared;

              if (aggregate) {
                  *shared = 0;
              }

              unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

              for (; i < blockSize * blockSize; i += gridDim.x * blockDim.x) {

                  if (intersection[i]) {
                      unsigned int probeID = i / blockSize + probeBlock.startID;
                      unsigned int candidateID = i % blockSize + indexedBlock.startID;

                      unsigned int probeSize = probeCollection.sizes[probeID];
                      unsigned int candidateSize = indexedCollection.sizes[candidateID];

                      unsigned int minoverlap = Similarity::minoverlap(probeSize, candidateSize, threshold);

                      if (intersection[i] >= minoverlap) {
                          if (aggregate) {
                              atomicAdd(sharedCount, 1);
                          } else {
                              unsigned int pos = atomicAdd(count, 1);
                              result.pairs[pos] = hyset::structs::pair(probeID, candidateID);
                          }
                      }
                  }
              }
              __syncthreads();
              if (aggregate && threadIdx.x == 0) {
                  result.counts[blockIdx.x] = *sharedCount;
              }
          }
      }

  }
};


#endif // HYSET_VERIFICATION_HPP