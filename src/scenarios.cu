#include <hyset/hybrid_kernels.hpp>

template <typename Similarity, bool aggregate>
__global__ void hyset::algorithms::hybrid::kernels::scenario_1(hyset::collection::device_collection_wrapper indexedCollection,
                           hyset::collection::device_collection_wrapper probeCollection,
                           hyset::containers::device_candidates_wrapper candidates,
                           hyset::containers::device_result_wrapper result,
                           unsigned int* globalCount,
                           unsigned int maxSetSize,
                           double threshold)

}


template class <typename Similarity, bool aggregate>