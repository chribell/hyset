# HySet: A Set Similarity Framework Using GPUs

Abstract: Set similarity join is a fundamental operation used in a wide range of applications such as data mining, data cleaning and entity resolution. Existing methods proposed for set similarity join conform to a filter-verification framework where potential candidate pairs are generated in the filtering phase and then undergo a verification phase to output the final result. Several different kinds of filtering techniques have been proposed and techniques also differentiate in the manner they couple filtering with verification. However, it has been shown that no globally dominant technique exists. Depending on the dataset and query characteristics, each technique has its own strong and weak points. Based on these findings, the main contribution of this work is the development of a hybrid framework for the set similarity join operation for a single GPU-equipped machine setting. Our framework  encapsulates a partitioning mechanism to utilize appropriately both the CPU and the GPU. We present all technical details and we show performance speedups up to 3.25x after thorough evaluation. 

# Compile

```
mkdir release && cd release
cmake .. -DCMAKE_BUILD_TYPE=Release -DSM_ARCH=61 # for Compute Capability 6.1
make -j 4
```
# Datasets

The datasets and the preprocess scripts can be found at http://ssjoin.dbresearch.uni-salzburg.at/datasets.html
