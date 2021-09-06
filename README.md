# BayesNetworkStructureLearning

In 'main', set the number of examples (n) the dataset should have and the beam size (b) of Beam Search. Then select one of the 3 available Bayesian Networks to learn its structure (comment out all but one). 
The dataset will then be generated, and the structure of the selected Bayesian Network will be attempted to be learned and then compared to the original structure.

If one wishes to test the accuracy of the result for multiple dataset sizes at once, simply use the provided code by specifying the number of tests (num_tests), the maximum dataset size (max_dataset_n) and finally the number of redundant tests (to smooth out the result) for each size (num_redundancy_tests). The result will be plotted.
