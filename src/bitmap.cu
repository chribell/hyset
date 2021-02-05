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
#include <hyset/device_algorithms.hpp>
#include <hyset/output.hpp>


typedef std::pair<hyset::structs::partition*, hyset::structs::partition*> partition_pair;
typedef std::map<partition_pair*, std::shared_ptr<hyset::output::handler>> output_map;
typedef std::map<std::string, std::vector<partition_pair>> algorithm_map;

int main(int argc, char** argv) {
    try {
        double threshold = 0.95;
        unsigned int blockSize = 10000;
        unsigned int bitmap = 64;
        std::string output;
        bool timings = false;

        cxxopts::Options options(argv[0], "HySet Framework: Set Similarity Join using GPUs");

        options.add_options()
                ("input", "Input dataset file", cxxopts::value<std::string>())
                ("foreign-input", "Foreign input dataset file", cxxopts::value<std::string>())
                ("block", "Block Size", cxxopts::value<unsigned int>(blockSize))
                ("threshold", "Similarity threshold", cxxopts::value<double>(threshold))
                ("bitmap", "Bitmap signature size", cxxopts::value<unsigned int>(bitmap))
                ("output", "Output file path", cxxopts::value<std::string>(output))
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
                "└{13:─^{1}}┘\n", "Arguments", 51, 25,
                "Input", input.c_str(),
                "Threshold", threshold,
                "Block", blockSize,
                "Bitmap", bitmap,
                "Output", output.c_str(), ""
        );

        hyset::timer::host timer;
        std::shared_ptr<hyset::timer::host> hostTimer = std::make_shared<hyset::timer::host>();
        std::shared_ptr<hyset::timer::device> deviceTimer = std::make_shared<hyset::timer::device>();

        hyset::collection::host_collection hostCollection;

        hyset::timer::host::Interval* readInput = hostTimer->add("Read input collection");
        std::vector<hyset::structs::block> blocks =
                hyset::collection::read_collection<jaccard>(input, hostCollection, blockSize, threshold);
        hostTimer->finish(readInput);

        hyset::timer::host::Interval* totalTime = timer.add("Total time");

        hyset::timer::device::EventPair* transferInput = deviceTimer->add("Transfer collection", 0);
        hyset::collection::device_collection deviceCollection(hostCollection);
        deviceTimer->finish(transferInput);

        std::vector<hyset::structs::block>::iterator block;

        hyset::structs::partition partition(0);
        for(block = blocks.begin();
            block != blocks.end();
            ++block) {
            partition.blocks.push_back(&(*block));
        }

        output_map outputHandlers;

        partition_pair pair;

        pair = partition_pair(&partition, &partition); // R join R

        if (output.empty()) {
            outputHandlers.insert(std::pair<partition_pair* , std::shared_ptr<hyset::output::count_handler>>(&pair, std::make_shared<hyset::output::count_handler>(hyset::output::count_handler())));
        } else {
            outputHandlers.insert(std::pair<partition_pair* , std::shared_ptr<hyset::output::pairs_handler>>(&pair, std::make_shared<hyset::output::pairs_handler>(hyset::output::pairs_handler())));
        }

        std::shared_ptr<hyset::collection::host_collection> hostCollectionPtr(&hostCollection);

        hyset::timer::host::Interval* joinTime = hostTimer->add("Join time");

        auto* bitmapHandler =
                new hyset::algorithms::device::bitmap::handler(deviceTimer,
                                                                hostCollectionPtr,
                                                                deviceCollection,
                                                                bitmap,
                                                                blockSize,
                                                                output.empty(),
                                                                threshold);
        bitmapHandler->setOutputHandler(outputHandlers[&pair]);
        bitmapHandler->join(pair.first, pair.second);

        hostTimer->finish(joinTime);


        timer.finish(totalTime);

        if (timings) {
            hostTimer->print();
            deviceTimer->print();
        }

        fmt::print("┌{0:─^{1}}┐\n"
                   "|{2: ^{1}}|\n"
                   "└{3:─^{1}}┘\n", "Total time without I/O (ms)", 51, timer.total(), "");

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

        return 0;
    } catch (const cxxopts::OptionException& e) {
        fmt::print(std::cerr, "Error parsing options: {}\n", e.what());
        exit(1);
    }
}
