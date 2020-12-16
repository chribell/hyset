#pragma once

#ifndef HYSET_PARTITIONER_HPP
#define HYSET_PARTITIONER_HPP

#include <hyset/structs.hpp>
#include <vector>

namespace hyset {
namespace partitioner {
    inline std::vector<hyset::structs::partition> even(std::vector<hyset::structs::block>& blocks, unsigned int numOfPartitions) {

        std::vector<hyset::structs::partition> partitions;
        unsigned int partitionSize = std::ceil((double) blocks.size() / (double) numOfPartitions);
        unsigned int numOfBlocks = blocks.size();
        unsigned int id = 0;

        for (unsigned int i = 0; i < numOfBlocks; i += partitionSize) {
            hyset::structs::partition partition = hyset::structs::partition(id++);
            std::vector<hyset::structs::block>::iterator block;

            for(block = blocks.begin() + i;
                block != blocks.begin() + std::min(numOfBlocks, i + partitionSize);
                ++block) {
                partition.blocks.push_back(&(*block));
            }
            partitions.push_back(partition);
        }
        return partitions;
    }

    inline std::vector<hyset::structs::partition> dichotomize(std::vector<hyset::structs::block>& blocks, double split) {
        std::vector<hyset::structs::partition> partitions;

        unsigned int portion = std::ceil((split / 100) * blocks.size());

        std::vector<hyset::structs::block> small(blocks.begin(), blocks.begin() + portion);
        std::vector<hyset::structs::block> large(blocks.begin() + portion, blocks.end());

        std::vector<hyset::structs::block>::iterator block;

        hyset::structs::partition pComplete(0);
        for(block = blocks.begin();
            block != blocks.end();
            ++block) {
            pComplete.blocks.push_back(&(*block));
        }

        hyset::structs::partition pSmall(1);
        for(block = blocks.begin();
            block != blocks.begin() + portion;
            ++block) {
            pSmall.blocks.push_back(&(*block));
        }

        hyset::structs::partition pLarge(2);

        for(block = blocks.begin() + portion;
            block != blocks.end();
            ++block) {
            pLarge.blocks.push_back(&(*block));
        }

        partitions.push_back(pComplete);
        partitions.push_back(pSmall);
        partitions.push_back(pLarge);
        return partitions;
    }
};
};


#endif // HYSET_PARTITIONER_HPP