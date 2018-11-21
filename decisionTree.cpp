// copyright Stefan Adrian, Luca Istrate, Andrei Medar
#include "./decisionTree.h"  // NOLINT(build/include)
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <random>

using std::string;
using std::pair;
using std::vector;
using std::unordered_map;
using std::make_shared;

Node::Node() {
    is_leaf = false;
    left = nullptr;
    right = nullptr;
}

void Node::make_decision_node(const int index, const int val) {
    split_index = index;
    split_value = val;
}

void Node::make_leaf(const vector<vector<int>> &samples,
                     const bool is_single_class) {
    // set the node as a leaf type
    // is_single_class = true -> all test have the same class(digit)
    // is_single_class = false -> the majority class is chosen

    is_leaf = true;
    if (is_single_class) {
    	result = samples[0][0];
    } else {
    	vector<int> freq(10, 0);
    	int size = samples.size();
    	for (int i = 0; i < size; ++i) {
    		++freq[samples[i][0]];
    	}

    	int max = freq[0];
    	int index = 0;
    	for (int i = 1; i < 10; ++i) {
    		if (freq[i] > max) {
    			max = freq[i];
    			index = i;
    		}
    	}

    	result = index;
    }
}

pair<int, int> find_best_split(const vector<vector<int>> &samples,
                               const vector<int> &dimensions) {
    // return the best split
    // the best split is the one that maximize information gain
    int splitIndex = -1, splitValue = -1;
	int size_d = dimensions.size();
	vector<int> uniqueValues;
	float max_IG = 0;
	float H_parent = get_entropy(samples);
	pair<vector<int>, vector<int>> subtrees;

	for (int i = 0; i < size_d; ++i) {
		uniqueValues = compute_unique(samples, dimensions[i]);
		int dim = uniqueValues.size();
		for (int j = 0; j < dim; ++j) {
			subtrees = get_split_as_indexes(samples, dimensions[i], uniqueValues[j]);

			if (!subtrees.first.size() || !subtrees.second.size()) {
				continue;
			}

			float left_entropy = get_entropy_by_indexes(samples, subtrees.first);
			float right_entropy = get_entropy_by_indexes(samples, subtrees.second);

			int left_size = subtrees.first.size();
			int right_size = subtrees.second.size();
			int n = left_size + right_size;

			float sum = left_size * left_entropy  + right_size * right_entropy;
			float IG = H_parent - sum / n;
			if (IG > max_IG) {
				max_IG = IG;
				splitIndex = dimensions[i];
				splitValue = uniqueValues[j];
			}
		}
	}

    return pair<int, int>(splitIndex, splitValue);
}

void Node::train(const vector<vector<int>> &samples) {
    // train the current node and his children, if it is necessary
    // 1). verify if all tests have the same class. if all tests have the
    // same class, then the node become a leaf.
    // 2). if there are not a valid split, the node become a leaf. Else,
    // the best split is choose and the code continues recursively.
    if (same_class(samples)) {
    	make_leaf(samples, true);
    } else {
    	vector<int> dimensions = random_dimensions(samples[0].size());
    	pair<int, int> best_split = find_best_split(samples, dimensions);
    	if (best_split.first == -1 && best_split.second == -1) {
    		make_leaf(samples, false);
    	} else {
    		pair<vector<vector<int>>, vector<vector<int>>> children;
    		children = split(samples, best_split.first, best_split.second);
    		make_decision_node(best_split.first, best_split.second);

    		left = make_shared<Node>(Node());
    		right = make_shared<Node>(Node());

    		left->train(children.first);
    		right->train(children.second);
    	}
    }
}

int Node::predict(const vector<int> &image) const {
    // return the predicted rezult by the decision tree
	if (is_leaf) {
		return result;
	}

	if (image[split_index - 1] <= split_value) {
		return left->predict(image);
	} else {
		return right->predict(image);
	}
}

bool same_class(const vector<vector<int>> &samples) {
    // check if all test have all the same class(digit)
	int first_class = samples[0][0];
	int size = samples.size();
    for (int i = 1; i < size; ++i) {
    	if (samples[i][0] != first_class) {
    		return false;
    	}
    }

    return true;
}

float get_entropy(const vector<vector<int>> &samples) {
    // return the tests entropie
    assert(!samples.empty());
    vector<int> indexes;

    int size = samples.size();
    for (int i = 0; i < size; i++) indexes.push_back(i);

    return get_entropy_by_indexes(samples, indexes);
}

float get_entropy_by_indexes(const vector<vector<int>> &samples,
                             const vector<int> &index) {
    // return the subset of tests entropie
	vector<int> freq(10, 0);
	int size = index.size();
	float H = 0;

	for (int i = 0; i < size; ++i) {
		++freq[samples[index[i]][0]];
	}

	for (int i = 0; i < 10; ++i) {
		if (freq[i]) {
			float p_i = (float)freq[i] / size;
			H += p_i * log2(p_i);
		}
	}

    return -H;
}

vector<int> compute_unique(const vector<vector<int>> &samples, const int col) {
    // return all the diffrent values that appear
    // in col column
    vector<int> uniqueValues;
    vector<int> freq(256, 0);
    int size = samples.size();

    for (int i = 0; i < size; ++i) {
    	freq[samples[i][col]] = 1;
    }

    for (int i = 0; i < 256; ++i) {
    	if (freq[i]) {
    		uniqueValues.push_back(i);
    	}
    }

    return uniqueValues;
}

pair<vector<vector<int>>, vector<vector<int>>> split(
    const vector<vector<int>> &samples, const int split_index,
    const int split_value) {
    // return the 2 subsets of tests obtained by separation
    // based on split_index and split_value
    vector<vector<int>> left, right;

    auto p = get_split_as_indexes(samples, split_index, split_value);
    for (const auto &i : p.first) left.push_back(samples[i]);
    for (const auto &i : p.second) right.push_back(samples[i]);

    return pair<vector<vector<int>>, vector<vector<int>>>(left, right);
}

pair<vector<int>, vector<int>> get_split_as_indexes(
    const vector<vector<int>> &samples, const int split_index,
    const int split_value) {
    // return the samples(atrrays) indexes from the two subsetes of tests
    // obtained by separation based on split_index and split_value
    vector<int> left, right;

    int size = samples.size();
    for (int i = 0; i < size; i++) {
    	if (samples[i][split_index] <= split_value) {
    		left.push_back(i);
    	} else {
    		right.push_back(i);
    	}
    }

    return make_pair(left, right);
}

vector<int> random_dimensions(const int size) {
    // return sqrt(size) different dimensions(indexes)
    std::uniform_int_distribution<int> d(0, size - 1);
    std::random_device rd;
    vector<int> rez;
    vector<int> freq(size, 0);

    int dim = floor(sqrt(size));

    for (int i = 0; i < dim; ++i) {
        int value = d(rd);

	    	if (value && !freq[value]) {
	    		freq[value] = 1;
	    		rez.push_back(value);
	    	} else {
	    		--i;
	    	}
    }

    return rez;
}
