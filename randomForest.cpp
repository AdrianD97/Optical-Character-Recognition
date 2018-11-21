// copyright Stefan Adrian, Luca Istrate, Andrei Medar
#include "randomForest.h"
#include <iostream>
#include <random>
#include <vector>
#include <string>
#include "decisionTree.h"

using std::vector;
using std::pair;
using std::string;
using std::mt19937;

vector<vector<int>> get_random_samples(const vector<vector<int>> &samples,
                                       int num_to_return) {
    // return an array with random elements
    std::uniform_int_distribution<int> d(0, num_to_return - 1);
    std::random_device rd;
    vector<vector<int>> ret;
    int size = samples.size();
    vector<int> freq(size, 0);

    for (int i = 0; i < num_to_return; ++i) {
        int value = d(rd);

        if (!freq[value]) {
            freq[value] = 1;
            ret.push_back(samples[value]);
        } else {
            --i;
        }
    }

    return ret;
}

RandomForest::RandomForest(int num_trees, const vector<vector<int>> &samples)
    : num_trees(num_trees), images(samples) {}

void RandomForest::build() {
    // alloc memory for each Tree
    // train each tree
    assert(!images.empty());
    vector<vector<int>> random_samples;

    int data_size = images.size() / num_trees;

    for (int i = 0; i < num_trees; i++) {
        random_samples = get_random_samples(images, data_size);

        // build a new Tree and train him
        trees.push_back(Node());
        trees[trees.size() - 1].train(random_samples);
    }
}

int RandomForest::predict(const vector<int> &image) {
    // return the most probable prediction for the test.
    // each tree is questioned and the final answer
    // is considered to be the majority.
    vector<int> freq(10, 0);

    for (int i = 0; i < num_trees; ++i) {
        ++freq[trees[i].predict(image)];
    }

    int max = -1;
    int c = -1;

    for (int i = 0; i < 10; ++i) {
        if (freq[i] > max) {
            max = freq[i];
            c = i;
        }
    }

    return c;
}
