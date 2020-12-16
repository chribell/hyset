#pragma once
#ifndef HYSET_SIMILARITY_HPP
#define HYSET_SIMILARITY_HPP

#include <hyset/helpers.hpp>
#include <cmath>
#define PMAXSIZE_EPS 1e-10

namespace hyset {
  namespace similarity {
      template <typename Similarity>
      struct generic_similarity {
          __forceinline__ __host__ __device__ static unsigned int maxprefix(unsigned int len, double threshold) {
              return hyset::min(len, len - minsize(len, threshold) + 1);
          }

          __forceinline__ __host__ __device__ static unsigned int midprefix(unsigned int len, double threshold) {
              return hyset::min(len, len - minoverlap(len, len, threshold) + 1);
          }

          __forceinline__ __host__ __device__ static unsigned int minoverlap(unsigned int len1, unsigned int len2, double threshold) {
              return hyset::min(len2, hyset::min(len1, Similarity::minoverlap(len1, len2, threshold)));
          }

          __forceinline__ __host__ __device__ static unsigned int minsize(unsigned int len, double threshold) {
              return Similarity::minsize(len, threshold);
          }

          __forceinline__ __host__ __device__ static unsigned int maxsize(unsigned int len, double threshold) {
              return Similarity::maxsize(len, threshold);
          }

          __forceinline__ __host__ __device__ static unsigned int maxsize(unsigned int len, unsigned int pos, double threshold) {
              return Similarity::maxsize(len, pos, threshold);
          }
      };

      struct jaccard_similarity {

          __forceinline__ __host__ __device__ static unsigned int minoverlap(unsigned int len1, unsigned int len2, double threshold) {
              return (unsigned int)(ceil((len1 + len2) * threshold / (1 + threshold)));
          }

          __forceinline__ __host__ __device__ static unsigned int minsize(unsigned int len, double threshold) {
              return (unsigned int)(ceil(threshold * len));
          }

          __forceinline__ __host__ __device__ static unsigned int maxsize(unsigned int len, double threshold) {
              return (unsigned int)((len / threshold));
          }

          __forceinline__ __host__ __device__ static unsigned int maxsize(unsigned int len, unsigned int pos, double threshold) {
              return (unsigned int)((len - ((1.0 - PMAXSIZE_EPS) + threshold) * pos) / threshold);
          }
      };
  };
};


typedef hyset::similarity::generic_similarity<hyset::similarity::jaccard_similarity> jaccard;

#endif // HYSET_SIMILARITY_HPP