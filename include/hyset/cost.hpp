#pragma once

#ifndef HYSET_COST_HPP
#define HYSET_COST_HPP

#include <hyset/structs.hpp>
#include <hyset/statistics.hpp>
#include <algorithm>
#include <map>
#include <string>


namespace hyset {

namespace cost {

    struct model {
        hyset::statistics::base stats;
        model(hyset::statistics::base& stats) : stats(stats) {}

        bool operator() (const std::pair<std::string, hyset::statistics::coefficients>& left,
                const std::pair<std::string, hyset::statistics::coefficients>& right ) {
            return left.second.cost(stats) < right.second.cost(stats);
        }
    };


    std::map<std::string, std::vector<std::pair<hyset::structs::partition*, hyset::structs::partition*>>>
        run(std::map<std::string, hyset::statistics::coefficients>& coefficients,
            std::vector<hyset::structs::partition>& partitions,
            hyset::collection::host_collection& hostCollection,
            hyset::collection::device_collection& deviceCollection,
            unsigned int bitmapSize) {

        std::map<std::string, std::vector<std::pair<hyset::structs::partition*, hyset::structs::partition*>>> map;

        std::vector<hyset::structs::partition>::iterator left;
        std::vector<hyset::structs::partition>::iterator right;

        for (left = partitions.begin(); left != partitions.end(); ++left) {
            hyset::statistics::base stats = hyset::statistics::extract(*left, hostCollection, deviceCollection);
            for (right = partitions.begin(); right != partitions.end(); ++right) {
                if (right->id < left->id) continue;
                if (left->id != right->id) {
                    stats = hyset::statistics::extract(*left, *right, hostCollection, deviceCollection);
                }
                stats.bitmap = bitmapSize;
                std::pair<std::string, hyset::statistics::coefficients> min = *std::min_element(coefficients.begin(), coefficients.end(), model(stats));

                if (map.count(min.first) == 0) {
                    map.insert(std::pair<std::string, std::vector<std::pair<hyset::structs::partition*, hyset::structs::partition*>>> (min.first, {}));
                }
                map[min.first].push_back(std::pair<hyset::structs::partition*, hyset::structs::partition*>(&(*left), &(*right)));
            }
        }

        return map;
    }
};

};



#endif // HOHYSET_COST_HPP