#pragma once

#ifndef HYSET_TIMER_HPP
#define HYSET_TIMER_HPP

#include <vector>
#include <string>
#include <chrono>
#include <iostream>
#include <iomanip>


namespace hyset {

namespace timer {
    struct host {
        struct Interval {
            typedef std::chrono::high_resolution_clock::time_point interval_point;
            interval_point begin;
            interval_point end;
            std::string name;
            Interval(const std::string& name) : name(name) {}
        };

        std::vector<Interval*> intervals;
        Interval* add(const std::string& name) {
            auto* interval = new Interval(name);
            interval->begin = std::chrono::high_resolution_clock::now();
            intervals.push_back(interval);
            return interval;
        }
        static void finish(Interval * interval) {
            interval->end = std::chrono::high_resolution_clock::now();
        }
        double sum(std::string const& name) const {
            double sum = 0.0;
            for(auto& interval : intervals) {
                if (interval->name == name) {
                    sum += (std::chrono::duration_cast<std::chrono::microseconds>
                                    (interval->end - interval->begin).count() / 1000.0);
                }
            }
            return sum;
        }
        double total() const {
            double total = 0.0;
            for(auto& interval : intervals) {
                total += (std::chrono::duration_cast<std::chrono::microseconds>
                                  (interval->end - interval->begin).count() / 1000.0);
            }
            return total;
        }
        void print() {
            fmt::print("┌{0:─^{1}}┐\n", "Host Timings (in ms)", 51);
            std::vector<std::string> distinctNames;
            for(auto& interval : intervals) {
                if (std::find(distinctNames.begin(), distinctNames.end(), interval->name) == distinctNames.end())
                    distinctNames.push_back(interval->name);
            }
            for(auto& name : distinctNames) {
                fmt::print("│{1: ^{0}}|{2: ^{0}}│\n", 25, name, sum(name));
            }
            fmt::print("└{1:─^{0}}┘\n", 51, "");
            fmt::print("│{1: ^{0}}|{2: ^{0}}│\n", 25, "Total", total());
            fmt::print("└{1:─^{0}}┘\n", 51, "");
        }
        ~host() {
            auto it = intervals.begin();
            for(; it != intervals.end(); ++it) delete *it;
        }
    };
    struct device {
        struct EventPair {
            std::string name;
            cudaEvent_t start;
            cudaEvent_t end;
            cudaStream_t stream;
            EventPair(std::string const& argName, cudaStream_t const& argStream) : name(argName), stream(argStream)  {}
        };
        std::vector<EventPair*> pairs;
        EventPair* add(std::string const& name, cudaStream_t const& stream = 0) {
            auto pair = new EventPair(name, stream);
            cudaEventCreate(&(pair->start));
            cudaEventCreate(&(pair->end));

            cudaEventRecord(pair->start, stream);

            pairs.push_back(pair);
            return pair;
        }
        static void finish(EventPair* pair) {
            cudaEventRecord(pair->end, pair->stream);
        }
        float sum(std::string const &name) const {
            float total = 0.0;
            auto it = pairs.begin();
            for(; it != pairs.end(); ++it) {
                if ((*it)->name == name) {
                    float millis = 0.0;
                    cudaEventElapsedTime(&millis, (*it)->start, (*it)->end);
                    total += millis;
                }
            }
            return total;
        }
        float total() const {
            float total = 0.0;
            auto it = pairs.begin();
            for(; it != pairs.end(); ++it) {
                float millis = 0.0;
                cudaEventElapsedTime(&millis, (*it)->start, (*it)->end);
                total += millis;
            }
            return total;
        }
        void print() {
            fmt::print("┌{0:─^{1}}┐\n", "Device Timings (in ms)", 51);
            std::vector<std::string> distinctNames;
            for(auto& pair : pairs) {
                if (std::find(distinctNames.begin(), distinctNames.end(), pair->name) == distinctNames.end())
                    distinctNames.push_back(pair->name);
            }
            for(auto& name : distinctNames) {
                fmt::print("│{1: ^{0}}|{2: ^{0}}│\n", 25, name, sum(name));
            }
            fmt::print("└{1:─^{0}}┘\n", 51, "");
            fmt::print("│{1: ^{0}}|{2: ^{0}}│\n", 25, "Total", total());
            fmt::print("└{1:─^{0}}┘\n", 51, "");
        }
        ~device() {
            for (auto& pair : pairs) {
                cudaEventDestroy(pair->start);
                cudaEventDestroy(pair->end);
                delete pair;
            }
        };
    };
};
};


#endif // HHYSET_TIMER_HPP