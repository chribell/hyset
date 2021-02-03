#include <iostream>
#include <map>
#include <set>
#include <iomanip>
#include <cxxopts.hpp>
#include <nlohmann/json.hpp>
#include <fmt/core.h>
#include <hyset/collection.hpp>
#include <hyset/similarity.hpp>
#include <hyset/statistics.hpp>
#include <hyset/partitioner.hpp>
#include <hyset/cost.hpp>
#include <hyset/timer.hpp>
#include <hyset/host_algorithms.hpp>
#include <hyset/hybrid_algorithms.hpp>
#include <hyset/device_algorithms.hpp>
#include <hyset/output.hpp>


typedef std::pair<hyset::structs::partition*, hyset::structs::partition*> partition_pair;
typedef std::map<partition_pair*, std::shared_ptr<hyset::output::handler>> output_map;
typedef std::map<std::string, std::vector<partition_pair>> algorithm_map;
typedef std::vector<hyset::structs::partition*> partition_vector;
typedef std::set<hyset::structs::partition*> partition_set;

std::map<std::string, hyset::statistics::coefficients> read_coefficients(std::string& path, double threshold);

partition_set extract_partitions(const std::vector<std::string>& techniques, algorithm_map& map);

void print_runs(algorithm_map& runs);

int main(int argc, char** argv) {
    try {
        std::string coefficientFile;
        double threshold = 0.95;
        unsigned int blockSize = 10000;
        unsigned int bitmap = 0;
        unsigned int numOfPartitions = 2;
        unsigned int threads = 32;
        std::string algorithm = "allpairs";
        std::string output;
        std::string deviceMemory = "512M";
        unsigned int scenario = 1;
        bool timings = false;

        cxxopts::Options options(argv[0], "HySet Framework: Set Similarity Join using GPUs");

        options.add_options()
                ("coefficients", "Coefficients json file", cxxopts::value<std::string>())
                ("input", "Input dataset file", cxxopts::value<std::string>())
                ("foreign-input", "Foreign input dataset file", cxxopts::value<std::string>())
                ("block", "Block Size", cxxopts::value<unsigned int>(blockSize))
                ("threshold", "Similarity threshold", cxxopts::value<double>(threshold))
                ("partitions", "Number of partitions", cxxopts::value<unsigned int>(numOfPartitions))
                ("bitmap", "Bitmap signature size", cxxopts::value<unsigned int>(bitmap))
                ("output", "Output file path", cxxopts::value<std::string>(output))
                ("threads", "Threads per block for hybrid solution", cxxopts::value<unsigned int>(threads))
                ("device-memory", "Device memory for hybrid solution", cxxopts::value<std::string>(deviceMemory))
                ("algorithm", "CPU filtering algorithm (allpairs|ppjoin)", cxxopts::value<std::string>(algorithm))
                ("scenario", "Device kernel scenario for hybrid solution", cxxopts::value<unsigned int>(scenario))
                ("timings", "Display timings", cxxopts::value<bool>(timings))
                ("help", "Print help");

        auto result = options.parse(argc, argv);

        if (result.count("help"))
        {
            std::cout << options.help() << std::endl;
            exit(0);
        }

        if (!result.count("input"))
        {
            fmt::print(std::cerr, "ERROR: No input dataset given! Exiting...\n");
            exit(1);
        }

        std::string input = result["input"].as<std::string>();

        fmt::print(
                "┌{0:─^{1}}┐\n"
                "│{3: ^{2}}|{4: ^{2}}│\n"
                "│{5: ^{2}}|{6: ^{2}}│\n"
                "│{7: ^{2}}|{8: ^{2}}│\n"
                "│{9: ^{2}}|{10: ^{2}}│\n"
                "│{11: ^{2}}|{12: ^{2}}│\n"
                "│{13: ^{2}}|{14: ^{2}}│\n"
                "│{15: ^{2}}|{16: ^{2}}│\n"
                "│{17: ^{2}}|{18: ^{2}}│\n"
                "│{19: ^{2}}|{20: ^{2}}│\n"
                "│{21: ^{2}}|{22: ^{2}}│\n"
                "└{23:─^{1}}┘\n", "Arguments", 51, 25,
                "Input", input.c_str(),
                "Threshold", threshold,
                "Partitions", numOfPartitions,
                "Block", blockSize,
                "Algorithm", algorithm,
                "Device memory", deviceMemory,
                "Bitmap", bitmap,
                "Scenario", scenario,
                "Threads", threads,
                "Output", output.c_str(), ""
        );

        char scale = deviceMemory.back();
        deviceMemory.pop_back();

        hyset::memory_calculator<unsigned int> calculator = {std::stod(deviceMemory), scale};

        hyset::timer::host hostTimings;
        hyset::timer::device deviceTimings;
        hyset::collection::host_collection hostCollection;

        hyset::timer::host::Interval* readInput = hostTimings.add("Read input collection");
        std::vector<hyset::structs::block> blocks =
                hyset::collection::read_collection<jaccard>(input, hostCollection, blockSize, threshold);
        hostTimings.finish(readInput);

        hyset::timer::device::EventPair* transferInput = deviceTimings.add("Transfer input collection", 0);
        hyset::collection::device_collection deviceCollection(hostCollection);
        deviceTimings.finish(transferInput);

        output_map outputHandlers;

        if (!result.count("coefficients"))
        {
            fmt::print(std::cerr, "ERROR: No input coefficients json required for partitioning is given! Exiting...\n");
            exit(1);
        }
        coefficientFile = result["coefficients"].as<std::string>();
        std::map<std::string, hyset::statistics::coefficients> coefficients = read_coefficients(coefficientFile, threshold);

        if (bitmap == 0) {
            coefficients.erase("bitmap");
        }

        hyset::timer::host::Interval* costModel = hostTimings.add("Running cost model");

        std::vector<hyset::structs::partition> partitions = hyset::partitioner::even(blocks, numOfPartitions);
        hyset::structs::partition largestPartition =
            *std::max_element(partitions.begin(), partitions.end(), [](hyset::structs::partition& a, hyset::structs::partition& b) {
                return a.size() < b.size();
            });

        algorithm_map runs = hyset::cost::run(coefficients, partitions, hostCollection, deviceCollection, bitmap);

        hostTimings.finish(costModel);

        print_runs(runs);

        for (auto& r : runs) {
            for(auto& j : r.second) {
                if (output.empty()) {
                    outputHandlers.insert(std::pair<partition_pair* , std::shared_ptr<hyset::output::count_handler>>(&j, std::make_shared<hyset::output::count_handler>(hyset::output::count_handler())));
                } else {
                    outputHandlers.insert(std::pair<partition_pair* , std::shared_ptr<hyset::output::pairs_handler>>(&j, std::make_shared<hyset::output::pairs_handler>(hyset::output::pairs_handler())));
                }
            }
        }

        partition_set indexedPartitions = extract_partitions({"cpu", "hybrid"}, runs);

        std::vector<hyset::index::host_array_index*> hostIndices;
        hostIndices.resize(partitions.size());

        hyset::timer::host::Interval* indexTime = hostTimings.add("Indexing for CPU-based algorithms");

        // for each left partition build inverted index
        for (auto& p : indexedPartitions) {
            hostIndices[p->id] = hyset::index::make_inverted_index(*p, hostCollection);
        }

        hostTimings.finish(indexTime);

        hyset::timer::host::Interval* joinTime = hostTimings.add("Join time");

        // First, we run any hybrid join in order to minimize dead spaces between host and device invocations
        if (!runs["hybrid"].empty()) {
            auto* deviceHandler =
                    new hyset::algorithms::hybrid::device_handler(deviceCollection,
                                                                   deviceCollection,
                                                                   threads,
                                                                   scenario,
                                                                   output.empty(),
                                                                   largestPartition.size(),
                                                                   calculator.numberOfElements(),
                                                                   threshold);

            for(auto& pair : runs["hybrid"]) {
                deviceHandler->setOutputHandler(outputHandlers[&pair]);
                deviceHandler->setMaxSetSize(hostCollection.sizes[pair.second->end()->endID]);
                if (algorithm == "allpairs") {
                    hyset::algorithms::hybrid::allpairs<hyset::structs::partition, hyset::index::host_array_index, jaccard>(pair.first,
                                                                  hostCollection,
                                                                  hostIndices[pair.first->id],
                                                                  pair.second,
                                                                  hostCollection,
                                                                  deviceHandler,
                                                                  threshold);
                } else {
                    hyset::algorithms::hybrid::ppjoin<hyset::structs::partition, hyset::index::host_array_index, jaccard>(
                                                                pair.first,
                                                                hostCollection,
                                                                hostIndices[pair.first->id],
                                                                pair.second,
                                                                hostCollection,
                                                                deviceHandler,
                                                                threshold);
                }
            }
        }

        // run cpu standalone
        if (!runs["cpu"].empty()) {
            for(auto& pair : runs["cpu"]) {
                if (algorithm == "allpairs") {
                    hyset::algorithms::host::allpairs<hyset::structs::partition, hyset::index::host_array_index, jaccard>(pair.first,
                                                                hostCollection,
                                                                hostIndices[pair.first->id],
                                                                pair.second,
                                                                hostCollection,
                                                                outputHandlers[&pair],
                                                                threshold);
                } else {
                    hyset::algorithms::host::ppjoin<hyset::structs::partition, hyset::index::host_array_index, jaccard>(pair.first,
                                                                hostCollection,
                                                                hostIndices[pair.first->id],
                                                                pair.second,
                                                                hostCollection,
                                                                outputHandlers[&pair],
                                                                threshold);
                }
            }
        }

        std::shared_ptr<hyset::collection::host_collection> hostCollectionPtr(&hostCollection);

        // run bitmap
        if (!runs["bitmap"].empty()) {
            auto* bitmapHandler =
                    new hyset::algorithms::device::bitmap::handler(hostCollectionPtr,
                                                                   deviceCollection,
                                                                   bitmap,
                                                                   blockSize,
                                                                   output.empty(),
                                                                   threshold);
            for(auto& pair : runs["bitmap"]) {
                bitmapHandler->setOutputHandler(outputHandlers[&pair]);
                bitmapHandler->join(pair.first, pair.second);
            }

        }

        // run fgssjoin
        if (!runs["fgssjoin"].empty()) {
            auto* fgssHandler =
                    new hyset::algorithms::device::fgssjoin::handler(hostCollectionPtr,
                                                                    deviceCollection,
                                                                    blockSize,
                                                                    output.empty(),
                                                                    threshold);
            for(auto& pair : runs["fgssjoin"]) {
                fgssHandler->setOutputHandler(outputHandlers[&pair]);
                fgssHandler->join(pair.first, pair.second);
            }
        }

        hostTimings.finish(joinTime);

        if (!output.empty()) { // output pairs to file
            fmt::print("Join finished, writing pairs to file\n");
            if (hyset::output::write_to_file(outputHandlers, output)) {
                fmt::print("Finished writing to file\n");
            } else {
                fmt::print("Error writing to file\n");
            }
        } else { // output count
            fmt::print("┌{0:─^{1}}┐\n"
                       "|{2: ^{1}}|\n"
                       "└{3:─^{1}}┘\n", "Result", 50, hyset::output::count(outputHandlers), "");
        }

        if (timings) {
            std::cout << hostTimings;
        }



        return 0;
    } catch (const cxxopts::OptionException& e) {
        fmt::print(std::cerr, "Error parsing options: {}\n", e.what());
        exit(1);
    }
}

std::map<std::string, hyset::statistics::coefficients> read_coefficients(std::string& path, double threshold)
{
    std::ostringstream strStream;
    strStream << std::setprecision(2) << threshold;
    std::string strThreshold = strStream.str();

    std::map<std::string, hyset::statistics::coefficients> coefficients;
    std::vector<std::string> techniques{"cpu", "hybrid", "bitmap", "fgssjoin"};

    std::ifstream file(path);
    nlohmann::json j = nlohmann::json::parse(file);

    for(auto& technique : techniques) {
        coefficients.insert( std::pair<std::string, hyset::statistics::coefficients>(technique, hyset::statistics::coefficients(
                j[technique][strThreshold]["Intercept"].get<long double>(),
                j[technique][strThreshold]["set_size_std"].get<long double>(),
                j[technique][strThreshold]["avg_set_size"].get<long double>(),
                j[technique][strThreshold]["token_freq"].get<long double>(),
                j[technique][strThreshold]["dataset_size"].get<long double>(),
                j[technique][strThreshold]["dataset_size_square"].get<long double>(),
                j[technique][strThreshold]["universe_size"].get<long double>(),
                (technique != "bitmap" ? j[technique][strThreshold]["index_size"].get<long double>() : 0),
                (technique == "bitmap" ? j[technique][strThreshold]["bitmap"].get<long double>() : 0)
        )));
    }

    return coefficients;
}


partition_set extract_partitions(const std::vector<std::string>& techniques, algorithm_map& map)
{
    partition_vector partitionVector;
    for(auto& technique : techniques) {
        for (auto& i : map[technique]) {
            partitionVector.push_back(i.first);
        }
    }
    return partition_set(partitionVector.begin(), partitionVector.end());
}

void print_runs(algorithm_map& runs)
{

    fmt::print("┌{0:─^{1}}┐\n", "Cost Model", 47);
    for(auto& run : runs) {
        for (auto& join : run.second) {
            fmt::print("│{1: ^{0}}|{2: ^{0}}|{3: ^{0}}│\n", 15, run.first, join.first->id, join.second->id);
        }
    }
    fmt::print("└{0:─^{1}}┘\n", "", 47);


}