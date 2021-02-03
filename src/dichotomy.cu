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
#include <hyset/host_algorithms.hpp>
#include <hyset/hybrid_algorithms.hpp>
#include <hyset/device_algorithms.hpp>
#include <hyset/output.hpp>


typedef std::pair<hyset::structs::partition*, hyset::structs::partition*> partition_pair;
typedef std::map<partition_pair*, std::shared_ptr<hyset::output::handler>> output_map;
typedef std::map<std::string, std::vector<partition_pair>> algorithm_map;

void print_runs(algorithm_map& runs);

std::vector<std::string> tokenize(std::string& str, char delimiter);

int main(int argc, char** argv) {
    try {
        double threshold = 0.95;
        unsigned int blockSize = 10000;
        unsigned int bitmap = 0;
        double split = 50.0;
        std::string algorithm = "allpairs";
        std::string policy = "cpu,gpu";
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
                ("split", "Blocks' split percentage", cxxopts::value<double>(split))
                ("bitmap", "Bitmap signature size", cxxopts::value<unsigned int>(bitmap))
                ("output", "Output file path", cxxopts::value<std::string>(output))
                ("algorithm", "CPU filtering algorithm (allpairs|ppjoin)", cxxopts::value<std::string>(algorithm))
                ("policy", "Running policy pair (cpu|gpu|cpu-gpu)", cxxopts::value<std::string>(policy))
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

        if (split > 100) {
            fmt::print(std::cerr, "ERROR: Wrong split percentage! Exiting...\n");
            exit(1);
        }

        std::vector<std::string> policies = tokenize(policy, ',');

        if (policies.size() != 2) {
            fmt::print(std::cerr, "ERROR: Wrong policy format! Exiting...\n");
            exit(1);
        }

        if ((policies[0] == policies[1]) ||
            (policies[0] == "cpu-gpu" && policies[1] == "gpu") ||
            (policies[0] == "gpu" && policies[1] == "cpu-gpu")) {
            fmt::print(std::cerr, "ERROR: Wrong policy values! Exiting...\n");
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
                "│{23: ^{2}}|{24: ^{2}}│\n"
                "└{25:─^{1}}┘\n", "Arguments", 51, 25,
                "Input", input.c_str(),
                "Threshold", threshold,
                "Block", blockSize,
                "Split(%)", split,
                "Policy", policy,
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

        hyset::timer::host timer;
        std::shared_ptr<hyset::timer::host> hostTimer = std::make_shared<hyset::timer::host>();
        std::shared_ptr<hyset::timer::device> deviceTimer = std::make_shared<hyset::timer::device>();
        hyset::collection::host_collection hostCollection;

        hyset::timer::host::Interval* readInput = hostTimer->add("Read input collection");
        std::vector<hyset::structs::block> blocks =
                hyset::collection::read_collection<jaccard>(input, hostCollection, blockSize, threshold);
        hostTimer->finish(readInput);

        hyset::timer::host::Interval* totalTime = timer.add("Total time");

        hyset::collection::host_collection hostCollectionCopy = hostCollection;

        hyset::timer::device::EventPair* transferInput = deviceTimer->add("Transfer collection", 0);
        hyset::collection::device_collection deviceCollection(hostCollection);
        deviceTimer->finish(transferInput);

        // should return three partitions (complete-whole dataset, split(%)-size, rest)
        std::vector<hyset::structs::partition> partitions = hyset::partitioner::dichotomize(blocks, split);

        fmt::print("┌{0:─^{1}}┐\n"
                   "|{2: ^{1}}|\n"
                   "└{3:─^{1}}┘\n", "Number of blocks", 51, blocks.size(), "");
        fmt::print(
                "┌{0:─^{1}}┐\n"
                "│{3: ^{2}}|{4: ^{2}}│\n"
                "│{5: ^{2}}|{6: ^{2}}│\n"
                "└{7:─^{1}}┘\n", "Blocks processed", 51, 25, "CPU",
                policies[0] == "cpu" ? partitions[1].blocks.size() : partitions[2].blocks.size(), "GPU",
                policies[0] != "cpu" ? partitions[1].blocks.size() : partitions[2].blocks.size(), "");

        output_map outputHandlers;

        hyset::index::host_array_index* mainIndex;
        hyset::index::host_array_index* threadIndex;

        hyset::timer::host::Interval* indexTime = hostTimer->add("Inverted index");

        if (policies[0] == "cpu") { // split-sized index
            mainIndex = hyset::index::make_inverted_index(partitions[1], hostCollection);
        } else { // complete index
            mainIndex = hyset::index::make_inverted_index(partitions[0], hostCollection);
        }

        if (policies[0] == "cpu-gpu") {
            threadIndex = hyset::index::make_inverted_index(partitions[1], hostCollection);
        }

        if (policies[1] == "cpu-gpu") {
            threadIndex = hyset::index::make_inverted_index(partitions[0], hostCollection);
        }

        hostTimer->finish(indexTime);

        std::string threadPolicy;
        partition_pair mainPair;
        partition_pair threadPair;

        if (policies[0] == "cpu") {
            mainPair = partition_pair(&partitions[1], &partitions[1]); // Rs join Rs
            threadPair = partition_pair (&partitions[0], &partitions[2]); // R join Rl
            threadPolicy = policies[1];
        } else {
            mainPair = partition_pair (&partitions[0], &partitions[2]); // R join Rl
            threadPair = partition_pair(&partitions[1], &partitions[1]); // Rs join Rs
            threadPolicy = policies[0];
        }

        if (output.empty()) {
            outputHandlers.insert(std::pair<partition_pair* , std::shared_ptr<hyset::output::count_handler>>(&mainPair, std::make_shared<hyset::output::count_handler>(hyset::output::count_handler())));
            outputHandlers.insert(std::pair<partition_pair* , std::shared_ptr<hyset::output::count_handler>>(&threadPair, std::make_shared<hyset::output::count_handler>(hyset::output::count_handler())));
        } else {
            outputHandlers.insert(std::pair<partition_pair* , std::shared_ptr<hyset::output::pairs_handler>>(&mainPair, std::make_shared<hyset::output::pairs_handler>(hyset::output::pairs_handler())));
            outputHandlers.insert(std::pair<partition_pair* , std::shared_ptr<hyset::output::pairs_handler>>(&threadPair, std::make_shared<hyset::output::pairs_handler>(hyset::output::pairs_handler())));
        }


        auto lambda = [&] {

            if (threadPolicy == "gpu") {
                std::shared_ptr<hyset::collection::host_collection> hostCollectionPtr(&hostCollectionCopy);
                if (bitmap > 0) { // bitmap
                    auto* bitmapHandler =
                            new hyset::algorithms::device::bitmap::handler(deviceTimer,
                                                                            hostCollectionPtr,
                                                                            deviceCollection,
                                                                            bitmap,
                                                                            blockSize,
                                                                            output.empty(),
                                                                            threshold);
                    bitmapHandler->setOutputHandler(outputHandlers[&threadPair]);
                    bitmapHandler->join(threadPair.first, threadPair.second);
                } else { // prefix
                    auto* fgssHandler =
                            new hyset::algorithms::device::fgssjoin::handler(deviceTimer,
                                                                              hostCollectionPtr,
                                                                              deviceCollection,
                                                                              blockSize,
                                                                              output.empty(),
                                                                              threshold);
                    fgssHandler->setOutputHandler(outputHandlers[&threadPair]);
                    fgssHandler->join(threadPair.first, threadPair.second);
                }
            } else {
                auto* deviceHandler =
                        new hyset::algorithms::hybrid::device_handler(deviceCollection,
                                                                       deviceCollection,
                                                                       threads,
                                                                       scenario,
                                                                       output.empty(),
                                                                       partitions[threadPair.first->id].size(),
                                                                       calculator.numberOfElements(),
                                                                       threshold);
                deviceHandler->setOutputHandler(outputHandlers[&threadPair]);
                deviceHandler->setMaxSetSize(hostCollection.sizes[threadPair.second->end()->endID]);
                if (algorithm == "allpairs") {
                    hyset::algorithms::hybrid::allpairs<hyset::structs::partition, hyset::index::host_array_index, jaccard>(threadPair.first,
                                                                                                                               hostCollectionCopy,
                                                                                                                               threadIndex,
                                                                                                                               threadPair.second,
                                                                                                                               hostCollectionCopy,
                                                                                                                               deviceHandler,
                                                                                                                               threshold);
                } else {
                    hyset::algorithms::hybrid::ppjoin<hyset::structs::partition, hyset::index::host_array_index, jaccard>(
                            threadPair.first,
                            hostCollectionCopy,
                            threadIndex,
                            threadPair.second,
                            hostCollectionCopy,
                            deviceHandler,
                            threshold);
                }
            }

        };


        std::thread deviceThread(lambda);

        if (algorithm == "allpairs") {
            hyset::algorithms::host::allpairs<hyset::structs::partition, hyset::index::host_array_index, jaccard>(
                    mainPair.first,
                    hostCollection,
                    mainIndex,
                    mainPair.second,
                    hostCollection,
                    outputHandlers[&mainPair],
                    threshold);
        } else {
            hyset::algorithms::host::ppjoin<hyset::structs::partition, hyset::index::host_array_index, jaccard>(
                    mainPair.first,
                    hostCollection,
                    mainIndex,
                    mainPair.second,
                    hostCollection,
                    outputHandlers[&mainPair],
                    threshold);
        }

        deviceThread.join();

        timer.finish(totalTime);

        if (timings) {
            hostTimer->print();
            deviceTimer->print();
        }

        fmt::print("┌{0:─^{1}}┐\n"
                   "|{2: ^{1}}|\n"
                   "└{3:─^{1}}┘\n", "Total time without I/O (secs)", 51, timer.total(), "");


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


        return 0;
    } catch (const cxxopts::OptionException& e) {
        fmt::print(std::cerr, "Error parsing options: {}\n", e.what());
        exit(1);
    }
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

std::vector<std::string> tokenize(std::string& str, char delimiter)
{
    std::vector<std::string> out;

    std::stringstream ss(str);
    std::string s;

    while (std::getline(ss, s, delimiter)) {
        out.push_back(s);
    }

    return out;
}