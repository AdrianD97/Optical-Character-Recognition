			# Optical Character Recognition, university project

	In this project i have developed a program that implements a Machine Learning algorithm (a slightly simplified version of the Random Forest algorithm) 
    that uses a decision tree to detect the digit in a 28x28 pixel image with up to 91% precision.
	The algorithm has two phases: 
		- the learning phase, in which the code recieve a series of images and have to learn to recognize them
		- the prediction phase, where the code recieve a series of images that have not been seen before and have to 
	    decide which digit is represented by the image.

	Each digit is represented as a 28x28 image. For simplicity, i have used a 784(28x28) length array to represent an image.
        Each pixel in the image is actually an integer from 0 to 255, the image being virtually gray-scale.

	Algorithm description

    OBS : class = digit
	  sample = array
	  dimension = index

    	Decision Tree
			- a decision tree is a binary tree. This is similar to a binary search tree. Unlike BST, in nodes that are not 
		leaves there are two values, a split index and a split value, that help us to decide in which direction to go.
		A leaf node contains the predicted class for the input array.
    	Learning
			- learning is done by dividing the data set by a certain rule, in a recursive way. When, from a division, 
		result a single class data set, then a leaf with that class is created. Splitting is done in the follow way: 
		for each dimension and for each value on that dimension a split is made. On each of these splits a metric is 
		applied which must be minimized/maximized. I have used Information Gain 
		metric(https://en.wikipedia.org/wiki/Information_gain_in_decision_trees).
    	Algorithm
			- the program choose the split that maximizes Information Gain, stores the split index and 
		the split value in that node, and then recursively go to the children node. If all the splits have a 
		child that has no element, then the code make a leaf node that contains the value of the majority class 
		of the data set. The splits are only forced on certain dimensions, randomly chosen. For each split, 
		the code first select sqrt(nr_dimensions) dimensions, then try splitting and maximize information gain only on those dimensions.

	The skeleton of code does not belong me. In this project, I have implememnted only the following functions/methods :
		- void make_leaf(const vector<vector<int>> &samples, const bool is_single_class)
		- bool same_class(const vector<vector<int>> &samples)
		- float get_entropy_by_indexes(const vector<vector<int>> &samples, const vector<int> &index)
		- vector<int> compute_unique(const vector<vector<int>> &samples, const int col)
		- pair<vector<int>, vector<int>> get_split_as_indexes(const vector<vector<int>> &samples, const int split_index, const int split_value)
		- vector<int> random_dimensions(const int size)
		- int predict(const vector<int> &image) const (from decisonTree.cpp)
		- int predict(const vector<int> &image) (from RandomForest.cpp)
		- vector<vector<int>> get_random_samples(const vector<vector<int>>& samples, int num_to_return)
		- pair<int, int> find_best_split(const vector<vector<int>> &samples, const vector<int> &dimensions)
		- void train(const vector<vector<int>> &samples)
