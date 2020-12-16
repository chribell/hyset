#pragma once

#ifndef HYSET_HOST_ALGORITHMS_HPP
#define HYSET_HOST_ALGORITHMS_HPP

#include <hyset/structs.hpp>
#include <hyset/containers.hpp>
#include <hyset/verification.hpp>
#include <hyset/output.hpp>


namespace hyset{
namespace algorithms {
namespace host {

    template <typename Segment, typename Index, typename Similarity>
    void allpairs(Segment* indexedSegment,
            hyset::collection::host_collection& indexedCollection,
            Index* invertedIndex,
            Segment* probeSegment,
            hyset::collection::host_collection& probeCollection,
            std::shared_ptr<hyset::output::handler>& output,
            double threshold) {
        hyset::containers::candidate_set candidateSet;

        std::vector<unsigned int> firstToCheck(indexedCollection.universeSize);
        std::vector<unsigned int> minoverlapCache;
        unsigned int lastProbeSize = 0;

        for (unsigned int probeID = probeSegment->start_id(); probeID <= probeSegment->end_id(); ++probeID) {
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


            // verify
            for (auto candit = candidateSet.begin(); candit != candidateSet.end(); ++candit) {
                hyset::structs::candidate_data& candidateData = indexedCollection.sets[*candit].candidateData;

                unsigned int indexedID = indexedCollection.sets[*candit].set_id;
                unsigned int indexedSize = indexedCollection.sizes[indexedID];

                unsigned int minoverlap = minoverlapCache[indexedSize];

                //First position after last position by index lookup in indexed record
                unsigned int lastposind = indexedCollection.sets[indexedID].prefixSize;

                //First position after last position by index lookup in probing record
                unsigned int lastposprobe = prefix;

                unsigned int recpreftoklast = probeCollection.sets[probeID].tokens[lastposprobe - 1];
                unsigned int indrecpreftoklast = indexedCollection.sets[indexedID].tokens[lastposind - 1];

                unsigned int recpos, indrecpos;

                if(recpreftoklast > indrecpreftoklast) {
                    recpos = candidateData.count;
                    //first position after minprefix / lastposind
                    indrecpos = lastposind;
                } else {
                    // First position after maxprefix / lastposprobe
                    recpos = lastposprobe;
                    indrecpos = candidateData.count;
                }

                if(hyset::verification::host::verify_pair(probeCollection.sets[probeID].tokens,
                              indexedCollection.sets[indexedID].tokens, minoverlap, recpos, indrecpos, candidateData.count)) {
                    output.get()->addPair(probeID, indexedID);
                }

                candidateData.reset();
            }

            candidateSet.clear();
        }
    }

    template <typename Segment, typename Index, typename Similarity>
    void ppjoin(Segment* indexedSegment,
                  hyset::collection::host_collection& indexedCollection,
                  Index* invertedIndex,
                  Segment* probeSegment,
                  hyset::collection::host_collection& probeCollection,
                  std::shared_ptr<hyset::output::handler>& output,
                  double threshold) {
        hyset::containers::candidate_set candidateSet;

        std::vector<unsigned int> firstToCheck(indexedCollection.universeSize);
        std::vector<unsigned int> minoverlapCache;
        unsigned int lastProbeSize = 0;
        for (unsigned int probeID = probeSegment->start_id(); probeID <= probeSegment->end_id(); ++probeID) {
            std::vector<unsigned int>& probeSet = probeCollection.sets[probeID].tokens;
            unsigned int probeSize = probeSet.size();
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
                unsigned int token = probeSet[pos];

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

                    unsigned int indexedPosition = ilit->position;
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

            // verify
            for (auto candit = candidateSet.begin(); candit != candidateSet.end(); ++candit) {
                hyset::structs::candidate_data& candidateData = indexedCollection.sets[*candit].candidateData;

                unsigned int indexedID = indexedCollection.sets[*candit].set_id;
                unsigned int indexedSize = indexedCollection.sizes[indexedID];

                unsigned int recpos = candidateData.probePosition;
                unsigned int indrecpos = candidateData.indexedPosition;

                //First position after last position by index lookup in indexed record
                unsigned int lastposind = indexedCollection.sets[indexedID].prefixSize;

                //First position after last position by index lookup in probing record
                unsigned int lastposprobe = prefix;

                unsigned int recpreftoklast = probeCollection.sets[probeID].tokens[lastposprobe - 1];
                unsigned int indrecpreftoklast = indexedCollection.sets[indexedID].tokens[lastposind - 1];

                if(recpreftoklast > indrecpreftoklast) {
                    recpos += 1;
                    //first position after minprefix / lastposind
                    indrecpos = lastposind;
                } else {
                    // First position after maxprefix / lastposprobe
                    recpos = lastposprobe;
                    indrecpos += 1;
                }

                if(hyset::verification::host::verify_pair(probeCollection.sets[probeID].tokens,
                                                          indexedCollection.sets[indexedID].tokens, candidateData.minoverlap, recpos, indrecpos, candidateData.count)) {
                    output.get()->addPair(probeID, indexedID);
                }

                candidateData.reset();
            }
            candidateSet.clear();
        }
    }

}
}
}



#endif // HYSET_HOST_ALGORITHMS_HPP