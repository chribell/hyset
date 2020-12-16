#pragma once

#ifndef HYSET_COLLECTION_HPP
#define HYSET_COLLECTION_HPP

#include <vector>
#include <tuple>
#include <cuda/api_wrappers.hpp>
#include <hyset/structs.hpp>
#include <hyset/similarity.hpp>
#include <hyset/containers.hpp>
#include <iostream>
#include <fstream>
#include <sstream>

std::vector<unsigned int> split(const std::string& s, char delimiter) {
    std::stringstream ss(s);
    std::string item;
    std::vector<unsigned int> elements;
    while (std::getline(ss, item, delimiter)) {
        elements.push_back(atoi(item.c_str()));
    }
    return elements;
}

namespace hyset {

namespace collection {

    struct host_collection {
        std::vector<hyset::structs::set> sets;
        std::vector<unsigned int> tokens;
        std::vector<hyset::structs::entry> prefixes;
        std::vector<unsigned int> sizes;
        std::vector<unsigned int> starts;
        std::vector<unsigned int> prefixStarts;
        unsigned int universeSize = 0;
        unsigned int maxEntries = 0;

        host_collection& operator = (const host_collection& hostCollection) {
            sets = hostCollection.sets;
            tokens = hostCollection.tokens;
            prefixes = hostCollection.prefixes;
            sizes = hostCollection.sizes;
            starts = hostCollection.starts;
            prefixStarts = hostCollection.prefixStarts;
            universeSize = hostCollection.universeSize;
            maxEntries = hostCollection.maxEntries;

            return *this;
        }
    };

    struct device_collection {
        hyset::containers::device_array<unsigned int>* tokens;
        hyset::containers::device_array<unsigned int>* starts;
        hyset::containers::device_array<unsigned int>* sizes;

        device_collection(hyset::collection::host_collection& collection) {
            tokens = new hyset::containers::device_array<unsigned int>(collection.tokens);
            starts = new hyset::containers::device_array<unsigned int>(collection.starts);
            sizes = new hyset::containers::device_array<unsigned int>(collection.sizes);
        }

        ~device_collection() {
            tokens->hyset::containers::device_array<unsigned int>::~device_array();
            starts->hyset::containers::device_array<unsigned int>::~device_array();
            sizes->hyset::containers::device_array<unsigned int>::~device_array();
        }

    };

    struct device_collection_wrapper {
        unsigned int* tokens;
        size_t tokensLen;
        unsigned int* starts;
        size_t startsLen;
        unsigned int* sizes;
        size_t sizesLen;

        device_collection_wrapper(device_collection& deviceCollection) :
            tokens(deviceCollection.tokens->ptr.get()),
            tokensLen(deviceCollection.tokens->length),
            starts(deviceCollection.starts->ptr.get()),
            startsLen(deviceCollection.starts->length),
            sizes(deviceCollection.sizes->ptr.get()),
            sizesLen(deviceCollection.sizes->length) {
        }

    };

    template <typename Similarity>
    std::vector<hyset::structs::block> read_collection(const std::string& filename,
            hyset::collection::host_collection& collection, unsigned int blockSize, double threshold) {
        std::vector<hyset::structs::block> blocks;

        std::ifstream infile;
        std::string line;
        infile.open(filename.c_str());

        if (!infile) {
            std::cerr << "Error: File not found, exiting...\n";
            exit(1);
        }

        unsigned int count = 0;
        unsigned int blockStartSetID = 0;

        unsigned int maxEntriesPerBlock = 0;
        unsigned int maxElement = 0;
        unsigned int setID = 0;
        unsigned int startSum = 0;
        unsigned int prefixStartSum = 0;
        unsigned int numOfTokens = 0;


        while (!infile.eof()) {
            std::getline(infile, line);
            if (line.empty()) continue;

            std::vector<unsigned int> tokens = split(line, ' ');
            unsigned int size = tokens.size();

            if (maxElement < tokens[tokens.size() - 1]) {
                maxElement = tokens[tokens.size() - 1];
            }

            unsigned int prefix = Similarity::midprefix(size, threshold);

            for (unsigned int i = 0; i < size; ++i) {
                collection.tokens.push_back(tokens[i]);
                if (i < prefix) {
                    collection.prefixes.push_back(hyset::structs::entry(tokens[i], setID, i));
                }
            }
            collection.sets.push_back(hyset::structs::set(setID, tokens, prefix));
            collection.sizes.push_back(size);
            collection.prefixStarts.push_back(prefixStartSum);
            collection.starts.push_back(startSum);
            startSum += size;
            prefixStartSum += prefix;
            numOfTokens += size;

            count++;
            setID++;

            if ( count == blockSize ) { // add new block
                count = 0;
                hyset::structs::block block = { (unsigned int) blocks.size(),
                                                 blockStartSetID,
                                                 setID - 1,
                                                 collection.prefixStarts[blockStartSetID],
                                                 prefixStartSum, numOfTokens  };
                blocks.push_back(block);
                blockStartSetID = setID;
                if (maxEntriesPerBlock < block.entries) {
                    maxEntriesPerBlock = block.entries;
                }
                numOfTokens = 0;
            }
        }
        if (count > 0) { // add final block
            hyset::structs::block block = { (unsigned int) blocks.size(),
                                             blockStartSetID,
                                             setID - 1,
                                             collection.prefixStarts[blockStartSetID],
                                             prefixStartSum, numOfTokens };
            blocks.push_back(block);
            if (maxEntriesPerBlock < block.entries) {
                maxEntriesPerBlock = block.entries;
            }
        }
        collection.universeSize = maxElement + 1; // add one for the zero token
        collection.maxEntries = maxEntriesPerBlock;
        infile.close();
        return blocks;
    }
};

};

#endif // HYSET_TIMER_HPP
