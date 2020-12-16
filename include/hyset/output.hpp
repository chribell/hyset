#pragma once

#ifndef HYSET_OUTPUT_HPP
#define HYSET_OUTPUT_HPP

#include <vector>
#include <hyset/structs.hpp>

#include <fmt/ostream.h>

namespace hyset{
namespace output {
    struct handler {
        virtual void addPair(const unsigned int& rec1, const unsigned int& rec2) = 0;
        virtual unsigned long getCount() const = 0;
        virtual ~handler() {}
    };

    struct count_handler : public handler {
        unsigned long count;
        count_handler() : count(0) {}
        void addPair(const unsigned int& rec1, const unsigned int& rec2) {
            ++count;
        }
        unsigned long getCount() const { return count; }
    };

    struct pairs_handler : public handler {
        std::vector<hyset::structs::pair> pairs;
        void addPair(const unsigned int& rec1, const unsigned int& rec2) {
            pairs.push_back(hyset::structs::pair(rec1, rec2));
        }
        const std::vector<hyset::structs::pair>& getPairs() const {
            return pairs;
        };
        unsigned long getCount() const { return pairs.size(); }
    };

    template <typename T>
    bool write_to_file(const std::map<T, std::shared_ptr<handler>>& outputHandlers, std::string& output) {
        std::ofstream file;
        file.open(output.c_str());

        if (file) {
            for(auto& x : outputHandlers) {
                auto* pairsHandler = (hyset::output::pairs_handler*) x.second.get();
                for (auto& pair : pairsHandler->pairs) {
                    fmt::print(file, "{} {}\n", pair.first, pair.second);
                }
            }

            file.close();
            return true;
        } else {
            return false;
        }
    }

    template <typename T>
    unsigned long count(const std::map<T, std::shared_ptr<handler>>& outputHandlers) {
        unsigned long count = 0;
        for(auto& x : outputHandlers) {
            count += x.second->getCount();
        }
        return count;
    }
}
};

#endif // HYSET_OUTPUT_HPP