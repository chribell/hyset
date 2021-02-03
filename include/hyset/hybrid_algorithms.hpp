#pragma once

#ifndef HYSET_HYBRID_ALGORITHMS_HPP
#define HYSET_HYBRID_ALGORITHMS_HPP

#include <thread>
#include <hyset/structs.hpp>
#include <hyset/containers.hpp>
#include <hyset/output.hpp>
#include <hyset/hybrid_kernels.hpp>


namespace hyset{
namespace algorithms {
namespace hybrid {
    struct device_handler {
        unsigned int scenario;
        unsigned int threads;
        std::thread deviceThread;
        bool aggregate;
        double threshold;
        unsigned maxSetSize;
        std::shared_ptr<hyset::output::handler> output;

        hyset::collection::device_collection_wrapper indexedCollection;
        hyset::collection::device_collection_wrapper probeCollection;


        hyset::containers::host_candidates* hostCandidates;
        hyset::containers::device_candidates* deviceCandidates;

        hyset::containers::device_result<unsigned int>* deviceCounts;
        hyset::containers::device_result<hyset::structs::pair>* devicePairs;

        cuda::memory::managed::unique_ptr<unsigned int> deviceCount;


        unsigned int maxProbes;
        unsigned int maxCandidates;

        device_handler(hyset::collection::device_collection& indexedCollection,
                hyset::collection::device_collection& probeCollection,
                unsigned int threads,
                unsigned int scenario,
                bool aggregate,
                unsigned int maxProbes,
                unsigned int maxCandidates,
                double threshold) :
            indexedCollection(indexedCollection), probeCollection(probeCollection), threads(threads),
            scenario(scenario), aggregate(aggregate), maxProbes(maxProbes),
            maxCandidates(maxCandidates), maxSetSize(0), threshold(threshold) {
            hostCandidates = new hyset::containers::host_candidates(maxProbes, maxCandidates);
            deviceCandidates = new hyset::containers::device_candidates(maxProbes, maxCandidates);
            if (aggregate) {
                deviceCounts = new hyset::containers::device_result<unsigned int>(maxCandidates);
            } else {
                devicePairs = new hyset::containers::device_result<hyset::structs::pair>(maxCandidates);
            }
            deviceCount = cuda::memory::managed::make_unique<unsigned int>();
            cuda::memory::device::set(deviceCount.get(), 0, sizeof(unsigned int));
        }

        inline void setMaxSetSize(unsigned int setSize) {
            maxSetSize = setSize;
        }

        inline void setOutputHandler(std::shared_ptr<hyset::output::handler>& outputHandler) {
            output = outputHandler;
        }

        inline void addCandidates(unsigned int probeID, hyset::containers::candidate_set& candidateSet) {

            // add probe set id
            hostCandidates->probes.push_back(probeID);

            if (candidateSet.size() + hostCandidates->candidates.size() >= maxCandidates) {

                unsigned int rest = maxCandidates - hostCandidates->candidates.size();

                hostCandidates->candidates.insert(std::end(hostCandidates->candidates),
                        std::begin(candidateSet.candidates),
                        std::begin(candidateSet.candidates) + rest);


                hostCandidates->offsets.push_back(hostCandidates->candidates.size());

                join();

                if (candidateSet.size() >  rest) {
                    hostCandidates->probes.push_back(probeID);
                    hostCandidates->candidates.insert(std::end(hostCandidates->candidates),
                                                      std::begin(candidateSet.candidates) + rest,
                                                      std::end(candidateSet.candidates));
                    hostCandidates->offsets.push_back(hostCandidates->candidates.size());

                }
            } else {
                hostCandidates->candidates.insert(std::end(hostCandidates->candidates),
                                                  std::begin(candidateSet.candidates),
                                                  std::end(candidateSet.candidates));

                hostCandidates->offsets.push_back(hostCandidates->candidates.size());
            }

        }

        inline void join() {
            // wait for device thread to finish (if it's running)
            if (deviceThread.joinable()) deviceThread.join();

            // transfer candidates to device
            deviceCandidates->copy(*hostCandidates);

            unsigned int probeCount = hostCandidates->probes.size();

            // clear host candidates to continue filtering
            hostCandidates->clear();

            deviceThread = std::thread(&hyset::algorithms::hybrid::device_handler::invoke, this, probeCount);
        }

        inline void invoke(unsigned int probeCount) {
            auto props = cuda::device::current::get().properties();
            unsigned int maxSharedMemory = cuda::device::detail::max_shared_memory_per_block(props.compute_capability());

            int gridX = probeCount;
            int threadsX = threads;

            unsigned int requiredSharedMemory;

            if (scenario == 1) {
                requiredSharedMemory = maxSetSize * sizeof(unsigned int) + (aggregate ? threads * sizeof(unsigned int) : 0);

                if (requiredSharedMemory > maxSharedMemory) {
                    maxSetSize -= (requiredSharedMemory - maxSharedMemory) / sizeof(unsigned int);
                    requiredSharedMemory = maxSetSize * sizeof(unsigned int) + (aggregate ? threads * sizeof(unsigned int) : 0);
                }
            } else {
                requiredSharedMemory = 2 * maxSetSize * sizeof(unsigned int) + threads * sizeof(unsigned int);

                if (requiredSharedMemory > maxSharedMemory) {
                    maxSetSize -= (requiredSharedMemory - maxSharedMemory) / sizeof(unsigned int);
                    requiredSharedMemory = 2 * maxSetSize * sizeof(unsigned int) + threads * sizeof(unsigned int);
                }
            }

            if (aggregate) {
                cuda::memory::zero({deviceCounts->output->ptr.get(), sizeof(unsigned int) * probeCount});

                if (scenario == 1) {
                    cuda::launch(hyset::algorithms::hybrid::kernels::scenario_1<jaccard, true>,
                                 cuda::launch_configuration_t(gridX, threadsX, requiredSharedMemory),
                                 probeCollection, probeCollection, *deviceCandidates, *deviceCounts, deviceCount.get(), maxSetSize, threshold
                    );
                } else {
                    cuda::launch(hyset::algorithms::hybrid::kernels::scenario_2<jaccard, true>,
                                 cuda::launch_configuration_t(gridX, threadsX, requiredSharedMemory),
                                 probeCollection, probeCollection, *deviceCandidates, *deviceCounts, deviceCount.get(), maxSetSize, threshold
                    );
                }

                thrust::device_ptr<unsigned int> thrustCount(deviceCounts->output->ptr.get());
                unsigned long result = thrust::reduce(thrust::device, thrustCount, thrustCount + probeCount);

                // downcast
                auto* countHandler = (hyset::output::count_handler*) output.get();
                countHandler->count += result;
            } else {
                if (scenario == 1) {
                    cuda::launch(hyset::algorithms::hybrid::kernels::scenario_1<jaccard, false>,
                                 cuda::launch_configuration_t(gridX, threadsX, requiredSharedMemory),
                                 probeCollection, probeCollection, *deviceCandidates, *devicePairs, deviceCount.get(), maxSetSize, threshold
                    );
                } else {
                    cuda::launch(hyset::algorithms::hybrid::kernels::scenario_2<jaccard, false>,
                                 cuda::launch_configuration_t(gridX, threadsX, requiredSharedMemory),
                                 probeCollection, probeCollection, *deviceCandidates, *devicePairs, deviceCount.get(), maxSetSize, threshold
                    );
                }

                std::unique_ptr<unsigned int> globalCount = std::unique_ptr<unsigned int>(new unsigned int);
                cuda::memory::copy(globalCount.get(), deviceCount.get(), sizeof(unsigned int));

                auto* pairsHandler = (hyset::output::pairs_handler*) output.get();
                std::vector<hyset::structs::pair> tmp;
                tmp.resize(*globalCount);
                cuda::memory::copy(&tmp[0], devicePairs->output->ptr.get(), sizeof(hyset::structs::pair) * (*globalCount));
                pairsHandler->pairs.insert(std::end(pairsHandler->pairs), std::begin(tmp), std::end(tmp));

                // we do not need to clear the pairs memory, just the global count in order for next pairs to be overwrite current data in future calls
                cuda::memory::device::zero(deviceCount.get(), sizeof(unsigned int));
            }
        }

        inline void flush() {
            if (!hostCandidates->candidates.empty()) {
                join();
                deviceThread.join();
            }
        }

        ~device_handler() {
            deviceCandidates->hyset::containers::device_candidates::~device_candidates();
        }
    };



    template <typename Segment, typename Index, typename Similarity>
    void allpairs(Segment* indexed,
            hyset::collection::host_collection& indexedCollection,
            Index* invertedIndex,
            Segment* probe,
            hyset::collection::host_collection& probeCollection,
            hyset::algorithms::hybrid::device_handler* deviceHandler,
            double threshold) {
        hyset::containers::candidate_set candidateSet;
        std::vector<unsigned int> firstToCheck(indexedCollection.universeSize);

        std::vector<unsigned int> minoverlapCache;
        unsigned int lastProbeSize = 0;

        for (unsigned int probeID = probe->start_id(); probeID <= probe->end_id(); ++probeID) {
            unsigned int probeSize = probeCollection.sizes[probeID];
            unsigned int probeStart = probeCollection.starts[probeID];
            unsigned int minsize = Similarity::minsize(probeSize, threshold);

            if (lastProbeSize != probeSize) {
                lastProbeSize = probeSize;
                unsigned int maxElement = probeSize;
                minoverlapCache.resize(maxElement + 1);
                for (unsigned int i = minsize; i <= maxElement; ++i) {
                    minoverlapCache[i] = Similarity::minoverlap(probeSize, i, threshold);
                }
            }

            unsigned int prefix = Similarity::maxprefix(probeSize, threshold);
            unsigned int maxsize = Similarity::maxsize(probeSize, threshold);

            // filter
            for (unsigned int pos = probeStart; pos < probeStart + prefix; ++pos) {
                unsigned int token = probeCollection.tokens[pos];

                // get iterator
                typename Index::iterator ilit = invertedIndex->get_iterator(token, firstToCheck[token]);

                // apply min-length filter
                while (!ilit.end()) {
                    unsigned int indexedID = ilit->setID;
                    unsigned int indexedSize = indexedCollection.sizes[indexedID];

                    if (ilit.length_filter(indexedSize, minsize)) { // check lower bound
                        break;
                    }
                }

                // for each record in inverted list
                while (!ilit.end()) {

                    unsigned int indexedID = ilit->setID;

                    if (indexedID >= probeID) {
                        break;
                    }
                    unsigned int indexedSize = indexedCollection.sizes[indexedID];
                    if (indexedSize > maxsize) {
                        break;
                    }

                    hyset::structs::candidate_data& candidateData = indexedCollection.sets[indexedID].candidateData;

                    if (candidateData.count == 0) {
                        candidateSet.addCandidate(indexedID);
                    }
                    candidateData.count += 1;
                    ilit.next();
                }

            }

            if (candidateSet.size() > 0) {
                deviceHandler->addCandidates(probeID, candidateSet);

                for (auto candit = candidateSet.begin(); candit != candidateSet.end(); ++candit) {
                    indexedCollection.sets[*candit].candidateData.reset();
                }
                candidateSet.clear();
            }
        }
        deviceHandler->flush();
    }

    template <typename Segment, typename Index, typename Similarity>
    void ppjoin(Segment* indexedPartition,
                hyset::collection::host_collection& indexedCollection,
                Index* invertedIndex,
                Segment* probePartition,
                hyset::collection::host_collection& probeCollection,
                hyset::algorithms::hybrid::device_handler* deviceHandler,
                double threshold) {
        hyset::containers::candidate_set candidateSet;
        std::vector<unsigned int> firstToCheck(indexedCollection.universeSize);

        std::vector<unsigned int> minoverlapCache;
        unsigned int lastProbeSize = 0;

        for (unsigned int probeID = probePartition->start_id(); probeID <= probePartition->end_id(); ++probeID) {
            unsigned int probeSize = probeCollection.sizes[probeID];
            unsigned int probeStart = probeCollection.starts[probeID];
            unsigned int minsize = Similarity::minsize(probeSize, threshold);

            if (lastProbeSize != probeSize) {
                lastProbeSize = probeSize;
                unsigned int maxElement = probeSize;
                minoverlapCache.resize(maxElement + 1);
                for (unsigned int i = minsize; i <= maxElement; ++i) {
                    minoverlapCache[i] = Similarity::minoverlap(probeSize, i, threshold);
                }
            }

            unsigned int prefix = Similarity::maxprefix(probeSize, threshold);
            unsigned int maxsize = Similarity::maxsize(probeSize, threshold);

            // filter
            for (unsigned int pos = 0; pos < prefix; ++pos) {
                unsigned int token = probeCollection.tokens[probeStart + pos];

                // get iterator
                typename Index::iterator ilit = invertedIndex->get_iterator(token, firstToCheck[token]);

                // apply min-length filter
                while (!ilit.end()) {
                    unsigned int indexedID = ilit->setID;
                    unsigned int indexedSize = indexedCollection.sizes[indexedID];

                    if (ilit.length_filter(indexedSize, minsize)) { // check lower bound
                        break;
                    }
                }

                while (!ilit.end()) {
                    unsigned int indexedID = ilit->setID;

                    if (indexedID >= probeID) {
                        break;
                    }

                    unsigned int indexedSize = indexedCollection.sizes[indexedID];
                    if (indexedSize > maxsize) {
                        break;
                    }

                    unsigned int indexedPosition = ilit->position;;
                    hyset::structs::candidate_data& candidateData = indexedCollection.sets[indexedID].candidateData;

                    if (candidateData.count == 0) {
                        unsigned int minoverlap = minoverlapCache[indexedSize];

                        if (!hyset::position_filter(probeSize, indexedSize, pos, indexedPosition, minoverlap)) {
                            ilit.next();
                            continue;
                        }

                        candidateSet.addCandidate(indexedID);
                        candidateData.minoverlap = minoverlapCache[indexedSize];
                    }

                    candidateData.count += 1;
                    candidateData.probePosition = pos;
                    candidateData.indexedPosition = indexedPosition;

                    ilit.next();
                }
            }

            if (candidateSet.size() > 0) {
                deviceHandler->addCandidates(probeID, candidateSet);

                for (auto candit = candidateSet.begin(); candit != candidateSet.end(); ++candit) {
                    indexedCollection.sets[*candit].candidateData.reset();
                }
                candidateSet.clear();
            }

        }
        deviceHandler->flush();
    }
}
}
}



#endif // HYSET_HYBRID_ALGORITHMS_HPP