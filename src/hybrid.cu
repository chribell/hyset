#include <iostream>
#include <map>
#include <set>
#include <cxxopts.hpp>
#include <fmt/core.h>
#include <hyset/collection.hpp>
#include <hyset/similarity.hpp>
#include <hyset/statistics.hpp>
#include <hyset/partitioner.hpp>
#include <hyset/timer.hpp>
#include <hyset/hybrid_algorithms.hpp>
#include <hyset/output.hpp>


typedef std::pair<hyset::structs::partition*, hyset::structs::partition*> partition_pair;
typedef std::map<partition_pair*, std::shared_ptr<hyset::output::handler>> output_map;
typedef std::map<std::string, std::vector<partition_pair>> algorithm_map;

int main(int argc, char** argv) {
    try {
        double threshold = 0.95;
        unsigned int blockSize = 10000;
        std::string algorithm = "allpairs";
        unsigned int threads = 32;
        std::string output;
        std::string deviceMemory = "512M";
        unsigned int scenario = 1;
        bool timings = false;

        cxxopts::Options options(argv[0], "HySet Framework: Set Similarity Join using GPUs");

        options.add_options()
                ("input", "Input dataset file", cxxopts::value<std::string>())
                ("foreign-input", "Foreign input dataset file", cxxopts::value<std::string>())
                ("block", "Block Size", cxxopts::value<unsigned int>(blockSize))
                ("threshold", "Similarity threshold", cxxopts::value<double>(threshold))
                ("output", "Output file path", cxxopts::value<std::string>(output))
                ("algorithm", "CPU filtering algorithm (allpairs|ppjoin)", cxxopts::value<std::string>(algorithm))
                ("threads", "Threads per block for hybrid solution", cxxopts::value<unsigned int>(threads))
                ("device-memory", "Device memory for hybrid solution", cxxopts::value<std::string>(deviceMemory))
                ("scenario", "Device kernel scenario for hybrid solution", cxxopts::value<unsigned int>(scenario))
                ("timings", "Display timings", cxxopts::value<bool>(timings))
                ("help", "Print help");

        auto result = options.parse(argc, argv);

        if (result.count("help"))
        {
            fmt::print("{}\n", options.help());
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
                "└{19:─^{1}}┘\n", "Arguments", 51, 25,
                "Input", input.c_str(),
                "Threshold", threshold,
                "Block", blockSize,
                "Algorithm", algorithm,
                "Device memory", deviceMemory,
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
        std::shared_ptr<hyset::collection::device_collection> deviceCollection = std::make_shared<hyset::collection::device_collection>(hostCollection);
        deviceTimings.finish(transferInput);


        std::vector<hyset::structs::block>::iterator block;

        hyset::structs::partition partition(0);
        for(block = blocks.begin();
            block != blocks.end();
            ++block) {
            partition.blocks.push_back(&(*block));
        }

        output_map outputHandlers;

        hyset::index::host_array_index* hostIndex;

        hyset::timer::host::Interval* indexTime = hostTimings.add("Indexing for CPU partition");

        hostIndex = hyset::index::make_inverted_index(partition, hostCollection);

        hostTimings.finish(indexTime);

        partition_pair pair;

        pair = partition_pair(&partition, &partition); // R join R

        if (output.empty()) {
            outputHandlers.insert(std::pair<partition_pair* , std::shared_ptr<hyset::output::count_handler>>(&pair, std::make_shared<hyset::output::count_handler>(hyset::output::count_handler())));
        } else {
            outputHandlers.insert(std::pair<partition_pair* , std::shared_ptr<hyset::output::pairs_handler>>(&pair, std::make_shared<hyset::output::pairs_handler>(hyset::output::pairs_handler())));
        }

        hyset::timer::host::Interval* joinTime = hostTimings.add("Join time");


        auto* deviceHandler =
                new hyset::algorithms::hybrid::device_handler(deviceCollection,
                                                               deviceCollection,
                                                               threads,
                                                               scenario,
                                                               output.empty(),
                                                               partition.size(),
                                                               calculator.numberOfElements(),
                                                               threshold);
        deviceHandler->setOutputHandler(outputHandlers[&pair]);
        deviceHandler->setMaxSetSize(hostCollection.sizes[pair.second->end()->endID]);
        if (algorithm == "allpairs") {
            hyset::algorithms::hybrid::allpairs<hyset::structs::partition, hyset::index::host_array_index, jaccard>(pair.first,
                                                                                                                       hostCollection,
                                                                                                                       hostIndex,
                                                                                                                       pair.second,
                                                                                                                       hostCollection,
                                                                                                                       deviceHandler,
                                                                                                                       threshold);
        } else {
            hyset::algorithms::hybrid::ppjoin<hyset::structs::partition, hyset::index::host_array_index, jaccard>(
                    pair.first,
                    hostCollection,
                    hostIndex,
                    pair.second,
                    hostCollection,
                    deviceHandler,
                    threshold);
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
            hostTimings.print();
        }



        return 0;
    } catch (const cxxopts::OptionException& e) {
        fmt::print(std::cerr, "Error parsing options: {}\n", e.what());
        exit(1);
    }
}