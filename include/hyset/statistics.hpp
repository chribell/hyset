#pragma once

#ifndef HYSET_STATISTICS_HPP
#define HYSET_STATISTICS_HPP

#include <hyset/index.hpp>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <cmath>
#include <limits>


// structure used to accumulate the moments and other
// statistical properties encountered so far.
template <typename T>
struct summary_stats_data
{
    T sum;
    T n;
    T min;
    T max;
    T mean;
    T M2;
    T M3;
    T M4;

    // initialize to the identity element
    void initialize()
    {
        sum = n = mean = M2 = M3 = M4 = 0;
        min = std::numeric_limits<T>::max();
        max = std::numeric_limits<T>::min();
    }

    T variance()   { return M2 / (n - 1); }
    T variance_n() { return M2 / n; }
    T skewness()   { return std::sqrt(n) * M3 / std::pow(M2, (T) 1.5); }
    T kurtosis()   { return n * M4 / (M2 * M2); }
    T average()    { return sum / n; }
};

// stats_unary_op is a functor that takes in a value x and
// returns a variace_data whose mean value is initialized to x.
template <typename T>
struct summary_stats_unary_op
{
    __host__ __device__
    summary_stats_data<T> operator()(const T& x) const
    {
        summary_stats_data<T> result;
        result.sum  = x;
        result.n    = 1;
        result.min  = x;
        result.max  = x;
        result.mean = x;
        result.M2   = 0;
        result.M3   = 0;
        result.M4   = 0;

        return result;
    }
};

// summary_stats_binary_op is a functor that accepts two summary_stats_data
// structs and returns a new summary_stats_data which are an
// approximation to the summary_stats for
// all values that have been aggregated so far
template <typename T>
struct summary_stats_binary_op
        : public thrust::binary_function<const summary_stats_data<T>&,
                const summary_stats_data<T>&,
                summary_stats_data<T> >
{
    __host__ __device__
    summary_stats_data<T> operator()(const summary_stats_data<T>& x, const summary_stats_data <T>& y) const
    {
        summary_stats_data<T> result;

        // precompute some common subexpressions
        T n  = x.n + y.n;
        T n2 = n  * n;
        T n3 = n2 * n;

        T delta  = y.mean - x.mean;
        T delta2 = delta  * delta;
        T delta3 = delta2 * delta;
        T delta4 = delta3 * delta;

        //Basic number of samples (n), min, and max
        result.n   = n;
        result.sum = x.sum + y.sum;
        result.min = thrust::min(x.min, y.min);
        result.max = thrust::max(x.max, y.max);

        result.mean = x.mean + delta * y.n / n;

        result.M2  = x.M2 + y.M2;
        result.M2 += delta2 * x.n * y.n / n;

        result.M3  = x.M3 + y.M3;
        result.M3 += delta3 * x.n * y.n * (x.n - y.n) / n2;
        result.M3 += (T) 3.0 * delta * (x.n * y.M2 - y.n * x.M2) / n;

        result.M4  = x.M4 + y.M4;
        result.M4 += delta4 * x.n * y.n * (x.n * x.n - x.n * y.n + y.n * y.n) / n3;
        result.M4 += (T) 6.0 * delta2 * (x.n * x.n * y.M2 + y.n * y.n * x.M2) / n2;
        result.M4 += (T) 4.0 * delta * (x.n * y.M3 - y.n * x.M3) / n;

        return result;
    }
};


namespace hyset {
namespace statistics {
    struct base {
        float setSizeStd = 0;
        float averageSetSize = 0;
        float tokenFreqDistribution = 0;
        unsigned int size = 0;
        unsigned int universeSize = 0;
        unsigned int indexSize = 0;
        unsigned int bitmap = 0;
    };

    struct coefficients {
        long double intercept = 0;
        long double setSizeStd = 0;
        long double averageSetSize = 0;
        long double tokenFreqDistribution = 0;
        long double size = 0;
        long double sizeSquare = 0;
        long double universeSize = 0;
        long double indexSize = 0;
        long double bitmap = 0;

        coefficients(long double intercept, long double setSizeStd, long double averageSetSize,
                long double tokenFreqDistribution, long double size, long double sizeSquare,
                long double universeSize, long double indexSize, long double bitmap)
                : intercept(intercept), setSizeStd(setSizeStd), averageSetSize(averageSetSize),
                  tokenFreqDistribution(tokenFreqDistribution), size(size), sizeSquare(sizeSquare),
                  universeSize(universeSize), indexSize(indexSize), bitmap(bitmap)   {}

        inline long double cost(hyset::statistics::base stats) const {
            return std::abs(setSizeStd * stats.setSizeStd +
                            averageSetSize * stats.averageSetSize +
                            tokenFreqDistribution * stats.tokenFreqDistribution +
                            size * stats.size +
                            sizeSquare * (stats.size * stats.size) +
                            universeSize * stats.universeSize +
                            indexSize * stats.indexSize +
                            bitmap * stats.bitmap +
                            intercept);
        }
    };

    hyset::statistics::base extract(hyset::structs::partition& partition,
            hyset::collection::host_collection& hostCollection,
            hyset::collection::device_collection& deviceCollection) {

        hyset::statistics::base stats;
        summary_stats_unary_op<float>  unary_op;
        summary_stats_binary_op<float> binary_op;
        summary_stats_data<float>      init;

        init.initialize();

        unsigned int startID = partition.start()->startID;
        unsigned int endID = partition.end()->endID;

        thrust::device_ptr<unsigned int> thrustSizes(deviceCollection.sizes->ptr.get());

        summary_stats_data<float> result = thrust::transform_reduce(thrustSizes + startID,
                                                                    thrustSizes + endID, unary_op, init, binary_op);

        stats.size = (endID - startID) + 1;
        stats.averageSetSize = result.average();
        stats.setSizeStd = std::sqrt(result.variance_n());

        // clear summary_stats for token frequency distribution
        init.initialize();

        unsigned int start = partition.start()->firstEntryPosition;
        unsigned int end = partition.end()->lastEntryPosition;
        unsigned int length = end - start;

        thrust::device_vector<hyset::structs::entry> entries(&hostCollection.prefixes[start], &hostCollection.prefixes[start] + length);
        thrust::device_vector<unsigned int> lengths(hostCollection.universeSize);

        auto current_device = cuda::device::current::get();
        int gridX = current_device.get_attribute(cudaDevAttrMultiProcessorCount) * 16;
        int threadsX = current_device.get_attribute(cudaDevAttrMaxThreadsPerBlock) / 2;

        cuda::launch(hyset::index::histogram,
                     cuda::launch_configuration_t(gridX, threadsX),
                     thrust::raw_pointer_cast(&entries[0]), thrust::raw_pointer_cast(&lengths[0]), length
                );


        result = thrust::transform_reduce(lengths.begin(), lengths.end(), unary_op, init, binary_op);

        stats.tokenFreqDistribution = std::sqrt(result.variance_n());
        stats.indexSize = length;
        stats.universeSize = hostCollection.universeSize;

        return stats;
    }


    hyset::statistics::base extract(hyset::structs::partition& left,
            hyset::structs::partition& right,
            hyset::collection::host_collection& hostCollection,
            hyset::collection::device_collection& deviceCollection) {

        hyset::statistics::base stats;
        summary_stats_unary_op<float>  unary_op;
        summary_stats_binary_op<float> binary_op;
        summary_stats_data<float>      init;

        init.initialize();

        unsigned int leftStartID = left.start()->startID;
        unsigned int leftEndID = left.end()->endID;
        unsigned int leftSize = (leftEndID - leftStartID) + 1;

        unsigned int rightStartID = right.start()->startID;
        unsigned int rightEndID = right.end()->endID;
        unsigned int rightSize = (rightEndID - rightStartID) + 1;

        thrust::device_vector<unsigned int> sizes(leftSize + rightSize);

        thrust::device_ptr<unsigned int> sizesPtr(deviceCollection.sizes->ptr.get());

        thrust::copy(sizesPtr + leftStartID, sizesPtr + leftEndID, sizes.begin());
        thrust::copy(sizesPtr + rightStartID, sizesPtr + rightEndID, sizes.begin() + leftSize);

        summary_stats_data<float> result = thrust::transform_reduce(sizes.begin(),
                                                                    sizes.end(), unary_op, init, binary_op);

        stats.size = leftSize + rightSize;
        stats.averageSetSize = result.average();
        stats.setSizeStd = std::sqrt(result.variance_n());

        init.initialize();

        unsigned int leftStartEntry = left.start()->firstEntryPosition;
        unsigned int leftEndEntry = left.end()->lastEntryPosition;
        unsigned int leftEntries = leftEndEntry - leftStartEntry;

        unsigned int rightStartEntry = right.start()->firstEntryPosition;
        unsigned int rightEndEntry = right.end()->lastEntryPosition;
        unsigned int rightEntries = rightEndEntry - rightStartEntry;

        thrust::device_vector<hyset::structs::entry> entries(leftEntries + rightEntries);
        thrust::device_vector<unsigned int> lengths(hostCollection.universeSize);

        thrust::copy(&hostCollection.prefixes[leftStartEntry], &hostCollection.prefixes[leftStartEntry] + leftEntries, entries.begin());
        thrust::copy(&hostCollection.prefixes[rightStartEntry], &hostCollection.prefixes[rightStartEntry] + rightEntries, entries.begin() + leftEntries);

        auto current_device = cuda::device::current::get();
        int gridX = current_device.get_attribute(cudaDevAttrMultiProcessorCount) * 16;
        int threadsX = current_device.get_attribute(cudaDevAttrMaxThreadsPerBlock) / 2;

        cuda::launch(hyset::index::histogram,
                     cuda::launch_configuration_t(gridX, threadsX),
                     thrust::raw_pointer_cast(&entries[0]), thrust::raw_pointer_cast(&lengths[0]), leftEntries + rightEntries
        );

        result = thrust::transform_reduce(lengths.begin(), lengths.end(), unary_op, init, binary_op);

        stats.tokenFreqDistribution = std::sqrt(result.variance_n());

        stats.indexSize = leftEntries;
        stats.universeSize = hostCollection.universeSize;

        return stats;
    }
}
};


#endif // HHYSET_STATISTICS_HPP