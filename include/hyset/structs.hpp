#pragma once

#ifndef HYSET_STRUCTS_HPP
#define HYSET_STRUCTS_HPP

namespace hyset {

namespace structs {

    struct block {
        unsigned int id;
        unsigned int startID;
        unsigned int endID;
        unsigned int firstEntryPosition;
        unsigned int lastEntryPosition;
        unsigned int numOfTokens;
        unsigned int entries;
        unsigned int size;

        block() = default;

        block(unsigned int id,
              unsigned int startID,
              unsigned int endID,
              unsigned int firstEntryPosition,
              unsigned int lastEntryPosition,
              unsigned int numOfTokens)
                : id(id), startID(startID), endID(endID),
                  firstEntryPosition(firstEntryPosition), lastEntryPosition(lastEntryPosition), numOfTokens(numOfTokens)
        {
            size = (endID - startID) + 1;
            entries = lastEntryPosition - firstEntryPosition;
        }

        inline unsigned int start_id() const {
            return startID;
        }

        inline unsigned int end_id() const {
            return endID;
        }

        inline double average_set_size() const {
            return  (double) numOfTokens / (double) size;
        }

        inline unsigned int first_entry_position() const {
            return firstEntryPosition;
        }
        inline unsigned int last_entry_position() const {
            return lastEntryPosition;
        }

    };

    struct entry {
        unsigned int token;
        unsigned int setID;
        unsigned int position;
        entry() = default;
        entry(unsigned int token, unsigned int setID, unsigned int position) :
                token(token), setID(setID), position(position) {}
    };

    struct entry_comparator {
        __host__ __device__
        bool operator()(const entry& entry1, const entry& entry2) {
            if (entry1.token < entry2.token) return true;
            if (entry2.token < entry1.token) return false;

            // entries have same token, check their set ids
            return entry1.setID < entry2.setID;
        }
    };

    struct pair {
        unsigned int first;
        unsigned int second;
        pair() = default;
        __host__ __device__ pair(unsigned int first, unsigned int second) : first(first), second(second) {}
    };

    struct candidate_data {
        unsigned int count = 0;
        unsigned int minoverlap = 0;
        unsigned int probePosition = 0;
        unsigned int indexedPosition = 0;

        candidate_data() : count(0) {}

        inline void reset () {
            count = 0;
        }
    };

    struct set {
        unsigned int set_id;
        std::vector<unsigned int> tokens;
        unsigned int prefixSize;
        candidate_data candidateData;
        set(unsigned int set_id, std::vector<unsigned int>& tokens, unsigned int prefixSize):
            set_id(set_id), tokens(tokens), prefixSize(prefixSize) {}
    };

    struct partition {
        unsigned int id;
        std::vector<hyset::structs::block*> blocks;

        partition(unsigned int id) : id(id) {}

        inline hyset::structs::block* start() {
            return blocks[0];
        }

        inline hyset::structs::block* end() {
            return blocks.back();
        }

        inline unsigned int size() {
            return (end()->endID - start()->startID) + 1;
        }

        inline unsigned int start_id () {
            return start()->start_id();
        }

        inline unsigned int end_id () {
            return end()->end_id();
        }

        inline unsigned int first_entry_position() {
            return start()->firstEntryPosition;
        }
        inline unsigned int last_entry_position() {
            return end()->lastEntryPosition;
        }


    };
}
}


#endif // HHYSET_STRUCTS_HPP