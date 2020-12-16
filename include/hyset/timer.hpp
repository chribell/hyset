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
            typedef std::chrono::steady_clock::time_point interval_point;
            interval_point begin;
            interval_point end;
            std::string descriptor;
            Interval(const std::string& descriptor) : descriptor(descriptor) {}
        };

        std::vector<Interval*> intervals;
        Interval* add(const std::string & descriptor);
        double sum(std::string const& argName) const;
        double total() const;
        void finish(Interval * interval);
        ~host();
        friend std::ostream & operator<<(std::ostream & os, const hyset::timer::host& timing);
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
        EventPair* add(std::string const& argName, cudaStream_t const& argStream);
        float sum(std::string const& argName) const;
        float total() const;
        void finish(EventPair* pair);
        ~device();
        friend std::ostream & operator<<(std::ostream& os, const device& timer);
    };
};
};


#endif // HHYSET_TIMER_HPP