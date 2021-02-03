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
#include <concurrentqueue.h>


typedef std::pair<hyset::structs::partition*, hyset::structs::partition*> partition_pair;
typedef std::pair<hyset::structs::block*, hyset::structs::block*> block_pair;
typedef std::map<int, std::shared_ptr<hyset::output::handler>> output_map;

int main(int argc, char** argv) {
    try {
        double threshold = 0.95;
        unsigned int blockSize = 10000;
        unsigned int bitmap = 0;
        std::string algorithm = "allpairs";
        unsigned int threads = 32;
        std::string output;
        std::string deviceMemory = "512M";
        unsigned int scenario = 1;
        bool timings = false;
        bool cooperative = false;

        cxxopts::Options options(argv[0], "HySet Framework: Set Similarity Join using GPUs");

        options.add_options()
                ("input", "Input dataset file", cxxopts::value<std::string>())
                ("foreign-input", "Foreign input dataset file", cxxopts::value<std::string>())
                ("block", "Block Size", cxxopts::value<unsigned int>(blockSize))
                ("threshold", "Similarity threshold", cxxopts::value<double>(threshold))
                ("bitmap", "Bitmap signature size", cxxopts::value<unsigned int>(bitmap))
                ("output", "Output file path", cxxopts::value<std::string>(output))
                ("algorithm", "CPU filtering algorithm (allpairs|ppjoin)", cxxopts::value<std::string>(algorithm))
                ("cooperative", "Run cpu-gpu in a separate thread", cxxopts::value<bool>(cooperative))
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
                "│{19: ^{2}}|{20: ^{2}}│\n"
                "│{21: ^{2}}|{22: ^{2}}│\n"
                "└{23:─^{1}}┘\n", "Arguments", 51, 25,
                "Input", input.c_str(),
                "Threshold", threshold,
                "Block", blockSize,
                "Cooperative", cooperative,
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

        fmt::print("┌{0:─^{1}}┐\n"
                   "|{2: ^{1}}|\n"
                   "└{3:─^{1}}┘\n", "Number of blocks", 51, blocks.size(), "");

        output_map outputHandlers;

        hyset::timer::host::Interval* indexTime = hostTimer->add("Inverted index");

        std::vector<hyset::index::host_array_index*> hostIndices;

        for (auto& block : blocks) {
            hostIndices.push_back(hyset::index::make_inverted_index(block, hostCollection));
        }

        hostTimer->finish(indexTime);

        if (output.empty()) {
            outputHandlers.insert(std::pair<int, std::shared_ptr<hyset::output::count_handler>>(0, std::make_shared<hyset::output::count_handler>(hyset::output::count_handler())));
            outputHandlers.insert(std::pair<int, std::shared_ptr<hyset::output::count_handler>>(1, std::make_shared<hyset::output::count_handler>(hyset::output::count_handler())));
        } else {
            outputHandlers.insert(std::pair<int, std::shared_ptr<hyset::output::pairs_handler>>(0, std::make_shared<hyset::output::pairs_handler>(hyset::output::pairs_handler())));
            outputHandlers.insert(std::pair<int, std::shared_ptr<hyset::output::pairs_handler>>(1, std::make_shared<hyset::output::pairs_handler>(hyset::output::pairs_handler())));
        }

        moodycamel::ConcurrentQueue<block_pair> queue;

        unsigned int deviceJoins = 0;
        unsigned int hostJoins = 0;


        std::vector<hyset::structs::block>::iterator firstBlock;
        std::vector<hyset::structs::block>::iterator secondBlock;

        for (firstBlock = blocks.begin(); firstBlock != blocks.end(); ++firstBlock) {

            unsigned int indexedID = (*firstBlock).id;

            for (secondBlock = blocks.begin(); secondBlock != blocks.end(); ++secondBlock) {

                unsigned int probeID = (*secondBlock).id;

                if (probeID < indexedID) continue; // this should only apply in case of self join (one input collection)

                unsigned int lastIndexedSize    = hostCollection.sizes[(*firstBlock).endID];
                unsigned int firstProbeSize     = hostCollection.sizes[(*secondBlock).startID];

                if ( lastIndexedSize >= jaccard::minsize(firstProbeSize, threshold)) {
                    queue.enqueue(block_pair(&(*firstBlock), &(*secondBlock)));
                }
            }
        }


        auto lambda = [&] {
            // TODO: find a more elegant way
            std::shared_ptr<hyset::collection::host_collection> hostCollectionPtr;

            hyset::algorithms::device::bitmap::handler* bitmapHandler;
            hyset::algorithms::device::fgssjoin::handler* fgssHandler;
            hyset::algorithms::hybrid::device_handler* hybridHandler;

            int lastBlockID = -1;

            if (bitmap > 0) {
                hostCollectionPtr = std::make_shared<hyset::collection::host_collection>(hostCollectionCopy);

                bitmapHandler = new hyset::algorithms::device::bitmap::handler(
                        deviceTimer,
                        hostCollectionPtr,
                        deviceCollection,
                        bitmap,
                        blockSize,
                        output.empty(),
                        threshold);
                bitmapHandler->setOutputHandler(outputHandlers[1]);
            } else if (cooperative) { // cpu-gpu
                hybridHandler = new hyset::algorithms::hybrid::device_handler(deviceCollection,
                                                                               deviceCollection,
                                                                               threads,
                                                                               scenario,
                                                                               output.empty(),
                                                                               blockSize,
                                                                               calculator.numberOfElements(),
                                                                               threshold);
                hybridHandler->setOutputHandler(outputHandlers[1]);
                hybridHandler->setMaxSetSize(hostCollection.sizes.back());
            } else { // prefix
                hostCollectionPtr = std::make_shared<hyset::collection::host_collection>(hostCollectionCopy);
                fgssHandler = new hyset::algorithms::device::fgssjoin::handler(
                        deviceTimer,
                        hostCollectionPtr,
                        deviceCollection,
                        blockSize,
                        output.empty(),
                        threshold);
                fgssHandler->setOutputHandler(outputHandlers[1]);
            }

            block_pair pair;
            while (queue.try_dequeue(pair)) {
                if (bitmap > 0) {
                    bitmapHandler->join(pair.first, pair.second);
                } else if (cooperative) {
                    if (algorithm == "allpairs") {
                        hyset::algorithms::hybrid::allpairs<hyset::structs::block, hyset::index::host_array_index, jaccard>(pair.first,
                                                                                                                               hostCollectionCopy,
                                                                                                                               hostIndices[pair.first->id],
                                                                                                                               pair.second,
                                                                                                                               hostCollectionCopy,
                                                                                                                               hybridHandler,
                                                                                                                               threshold);
                    } else {
                        hyset::algorithms::hybrid::ppjoin<hyset::structs::block, hyset::index::host_array_index, jaccard>(
                                pair.first,
                                hostCollectionCopy,
                                hostIndices[pair.first->id],
                                pair.second,
                                hostCollectionCopy,
                                hybridHandler,
                                threshold);
                    }
                } else {
                    // copy host_array_index to device_index first
                    if (lastBlockID != pair.first->id) {
                        fgssHandler->transferIndex(hostIndices[pair.first->id]);
                        lastBlockID = pair.first->id;
                    }
                    // then join
                    fgssHandler->join(pair.first, pair.second);
                }
                deviceJoins++;
            }

        };


        std::thread deviceThread(lambda);
        block_pair pair;
        while (queue.try_dequeue(pair)) {
            hyset::timer::host::Interval* joinTime = hostTimer->add("Join");
            if (algorithm == "allpairs") {
                hyset::algorithms::host::allpairs<hyset::structs::block, hyset::index::host_array_index, jaccard>(
                        pair.first,
                        hostCollection,
                        hostIndices[pair.first->id],
                        pair.second,
                        hostCollection,
                        outputHandlers[0],
                        threshold);
            } else {
                hyset::algorithms::host::ppjoin<hyset::structs::block, hyset::index::host_array_index, jaccard>(
                        pair.first,
                        hostCollection,
                        hostIndices[pair.first->id],
                        pair.second,
                        hostCollection,
                        outputHandlers[0],
                        threshold);
            }
            hostTimer->finish(joinTime);
            hostJoins++;
        }

        deviceThread.join();

        timer.finish(totalTime);

        unsigned int totalJoins = hostJoins + deviceJoins;

        double hostPercentage = ((double) hostJoins / (double) totalJoins) * 100.0;
        double devicePercentage = ((double) deviceJoins / (double) totalJoins) * 100.0;


        fmt::print(
                "┌{0:─^{1}}┐\n"
                "│{4: ^{2}}|{5: ^{2}}|{6: ^{3}}│\n"
                "│{7: ^{2}}|{8: ^{2}}|{9: ^{3}}│\n"
                "└{10:─^{1}}┘\n", "Joins processed", 51, 16, 17, "CPU", hostJoins, std::to_string(hostPercentage) + "%", "GPU", deviceJoins, std::to_string(devicePercentage) + "%", "");

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
