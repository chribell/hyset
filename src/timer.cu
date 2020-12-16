#include <hyset/timer.hpp>
#include <algorithm>

hyset::timer::host::Interval* hyset::timer::host::add(const std::string & descriptor)
{
    auto* interval = new hyset::timer::host::Interval(descriptor);
    this->intervals.push_back(interval);
    interval->begin = std::chrono::steady_clock::now();
    return interval;
}

void hyset::timer::host::finish(hyset::timer::host::Interval* interval)
{
    interval->end = std::chrono::steady_clock::now();
}

double hyset::timer::host::sum(std::string const& argName) const {
    double sum = 0.0;
    for(auto& interval : intervals) {
        if (interval->descriptor == argName) {
            sum += std::chrono::duration_cast<std::chrono::microseconds>
                           (interval->end - interval->begin).count() / 1000000.0;
        }
    }
    return sum;
}


double hyset::timer::host::total() const {
    double total = 0.0;
    for(auto& interval : intervals) {
        total += std::chrono::duration_cast<std::chrono::microseconds>
                         (interval->end - interval->begin).count() / 1000000.0;
    }
    return total;
}

std::ostream & hyset::timer::operator<<(std::ostream& os, const hyset::timer::host& timer)
{
    std::vector<std::string> distinctNames;
    for(auto& interval : timer.intervals) {
        if (std::find(distinctNames.begin(), distinctNames.end(), interval->descriptor) == distinctNames.end())
            distinctNames.push_back(interval->descriptor);
    }
    for(auto& name : distinctNames) {
        os << name << std::setw(11) << timer.sum(name) << " secs" << std::endl;
    }

    return os;
}

hyset::timer::host::~host()
{
    auto it = intervals.begin();
    for(; it != intervals.end(); ++it) delete *it;
}

hyset::timer::device::EventPair* hyset::timer::device::add(const std::string & argName, cudaStream_t const& argStream) {

    auto pair = new EventPair(argName, argStream);

    cudaEventCreate(&(pair->start));
    cudaEventCreate(&(pair->end));

    cudaEventRecord(pair->start, argStream);

    pairs.push_back(pair);
    return pair;
}

void hyset::timer::device::finish(EventPair* pair) {
    cudaEventRecord(pair->end, pair->stream);
}

float hyset::timer::device::sum(std::string const &argName) const {
    float total = 0.0;
    auto it = pairs.begin();
    for(; it != pairs.end(); ++it) {
        if ((*it)->name == argName) {
            float millis = 0.0;
            cudaEventElapsedTime(&millis, (*it)->start, (*it)->end);
            total += millis;
        }
    }
    return total;
}

hyset::timer::device::~device() {
    auto it = pairs.begin();
    for(; it != pairs.end(); ++it) {
        cudaEventDestroy((*it)->start);
        cudaEventDestroy((*it)->end);
        delete *it;
    }
}

float hyset::timer::device::total() const {
    float total = 0.0;
    auto it = pairs.begin();
    for(; it != pairs.end(); ++it) {
        float millis = 0.0;
        cudaEventElapsedTime(&millis, (*it)->start, (*it)->end);
        total += millis;
    }
    return total;
}

std::ostream & hyset::timer::operator<<(std::ostream & os, const hyset::timer::device& timer) {
    std::vector<std::string> distinctNames;
    for(auto& pair : timer.pairs) {
        if (std::find(distinctNames.begin(), distinctNames.end(), pair->name) == distinctNames.end())
            distinctNames.push_back(pair->name);
    }
    for(auto& name : distinctNames) {
        os << name << std::setw(11) << timer.sum(name) << " ms" << std::endl;
    }

    return os;
}

