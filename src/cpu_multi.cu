#include <iostream>
#include <map>
#include <set>
#include <algorithm>
#include <cxxopts.hpp>
#include <fmt/core.h>
#include <hyset/collection.hpp>
#include <hyset/similarity.hpp>
#include <hyset/statistics.hpp>
#include <hyset/partitioner.hpp>
#include <hyset/timer.hpp>
#include <hyset/host_algorithms.hpp>
#include <hyset/output.hpp>
#include <thread>


typedef std::pair<hyset::structs::partition*, hyset::structs::partition*> partition_pair;
typedef std::map<partition_pair*, std::shared_ptr<hyset::output::handler>> output_map;
typedef std::map<std::string, std::vector<partition_pair>> algorithm_map;
typedef std::vector<hyset::structs::block>::iterator block_iterator;

int main(int argc, char** argv) {
    try {
        double threshold = 0.95;
        unsigned int blockSize = 10000;
        std::string algorithm = "allpairs";
        std::string output;
        bool timings = false;
        unsigned int numOfPartitions = 2;
        bool even = false;

        cxxopts::Options options(argv[0], "HySet Framework: Set Similarity Join using GPUs");

        options.add_options()
                ("input", "Input dataset file", cxxopts::value<std::string>())
                ("block", "Block Size", cxxopts::value<unsigned int>(blockSize))
                ("threshold", "Similarity threshold", cxxopts::value<double>(threshold))
                ("output", "Output file path", cxxopts::value<std::string>(output))
                ("algorithm", "CPU filtering algorithm (allpairs|ppjoin)", cxxopts::value<std::string>(algorithm))
                ("partitions", "Number of partitions to split input workload", cxxopts::value<unsigned int>(numOfPartitions))
                ("even", "Type of workload partitioning", cxxopts::value<bool>(even))
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
                "└{15:─^{1}}┘\n", "Arguments", 51, 25,
                "Input", input.c_str(),
                "Threshold", threshold,
                "Block", blockSize,
                "Algorithm", algorithm,
                "Partitions", numOfPartitions,
                "Output", output.c_str(), ""
        );

        hyset::timer::host timer;
        hyset::timer::host hostTimings;
        hyset::timer::device deviceTimings;
        hyset::collection::host_collection hostCollection;

        hyset::timer::host::Interval* readInput = hostTimings.add("Read input collection");
        std::vector<hyset::structs::block> blocks =
                hyset::collection::read_collection<jaccard>(input, hostCollection, blockSize, threshold);
        hyset::timer::host::finish(readInput);

        hyset::timer::host::Interval* totalTime = timer.add("Total time");

        hyset::timer::device::EventPair* transferInput = deviceTimings.add("Transfer input collection", 0);
        hyset::collection::device_collection deviceCollection(hostCollection);
        hyset::timer::device::finish(transferInput);

        std::vector<std::shared_ptr<hyset::structs::partition>> probePartitions;
        std::vector<std::shared_ptr<hyset::index::host_array_index>> indices;

        if (even) {
            unsigned int blocksPerPartition = blocks.size() / numOfPartitions;
            block_iterator globalIterator;

            globalIterator = blocks.begin();

            unsigned int iter = 0;

            for (unsigned int left = blocks.size(); left != 0; ) {
                auto const skip = std::min(left, blocksPerPartition);

                auto partition = std::make_shared<hyset::structs::partition>(iter);
                for (auto b = globalIterator; b != globalIterator + skip; b++) {
                    partition->blocks.push_back(&(*b));
                }
                probePartitions.push_back(partition);
                left -= skip;
                std::advance(globalIterator, skip);
                iter++;
            }
        } else {
            // x_i = round( k * sqrt((i + 1) / numOfPartitions))
            unsigned int k = hostCollection.sets.size();
            std::vector<unsigned int> points;
            points.push_back(0);

            for (unsigned int i = 0; i < numOfPartitions; ++i) {
                points.push_back(k * std::sqrt((double) (i + 1) / (double) numOfPartitions));
            }

            unsigned int iter = 0;
            for (unsigned int i = 0; i < points.size() - 1; ++i) {
                unsigned int startID = points[i];
                unsigned int endID = points[i + 1] - 1;
                probePartitions.push_back(
                        std::make_shared<hyset::structs::partition>(iter++,
                                                                    startID,
                                                                    endID,
                                                                    hostCollection.prefixStarts[startID],
                                                                    hostCollection.prefixStarts[endID] + jaccard::midprefix(hostCollection.sizes[endID] , threshold)
                                                                    )
                );
            }

        }

        std::vector<std::shared_ptr<hyset::structs::partition>> indexedPartitions;

        for (const auto& p : probePartitions) {
            indexedPartitions.push_back(
                    std::make_shared<hyset::structs::partition>(p->id, 0, p->end_id(), 0, p->last_entry_position())
            );
        }

        hyset::timer::host::Interval* indexTime = hostTimings.add("Indexing");

        for (const auto& p : indexedPartitions) {
            indices.push_back(
                    std::make_shared<hyset::index::host_array_index>(*hyset::index::make_inverted_index(*p, hostCollection))
            );
        }
        hyset::timer::host::finish(indexTime);


        std::vector<std::shared_ptr<hyset::output::handler>> outputHandlers;
        std::vector<hyset::collection::host_collection> hostCollectionCopies;

        for (unsigned int i = 0; i < numOfPartitions; ++i) {
            if (output.empty()) {
                outputHandlers.push_back(std::make_shared<hyset::output::count_handler>(hyset::output::count_handler()));
            } else {
                outputHandlers.push_back(std::make_shared<hyset::output::pairs_handler>(hyset::output::pairs_handler()));
            }

            // lazy fix to remove sharing same candidateData structs among threads
            hostCollectionCopies.push_back(hostCollection);
        }

        hyset::timer::host::Interval* joinTime = hostTimings.add("Join time");

        auto lambda = [&] (unsigned int idx) {
            if (algorithm == "allpairs") {
                hyset::algorithms::host::allpairs<hyset::structs::partition, hyset::index::host_array_index, jaccard>(
                        indexedPartitions[idx].get(),
                        hostCollectionCopies[idx],
                        indices[idx].get(),
                        probePartitions[idx].get(),
                        hostCollection,
                        outputHandlers[idx],
                        threshold);
            } else {
                hyset::algorithms::host::ppjoin<hyset::structs::partition, hyset::index::host_array_index, jaccard>(
                        indexedPartitions[idx].get(),
                        hostCollectionCopies[idx],
                        indices[idx].get(),
                        probePartitions[idx].get(),
                        hostCollection,
                        outputHandlers[idx],
                        threshold);
            }
        };

        std::vector<std::thread> threads;

        for (unsigned int i = 0; i < numOfPartitions; ++i) {
            threads.push_back(std::thread(lambda, i));
        }

        for (auto& t : threads) {
            t.join();
        }

        hyset::timer::host::finish(joinTime);

        hyset::timer::host::finish(totalTime);

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
                       "└{3:─^{1}}┘\n", "Result", 51, hyset::output::count(outputHandlers), "");
        }

        if (timings) {
            hostTimings.print();
        }

        fmt::print("┌{0:─^{1}}┐\n"
                   "|{2: ^{1}}|\n"
                   "└{3:─^{1}}┘\n", "Total time without I/O (ms)", 51, timer.total(), "");

        return 0;
    } catch (const cxxopts::OptionException& e) {
        fmt::print(std::cerr, "Error parsing options: {}\n", e.what());
        exit(1);
    }
}
