import numpy as np
from collections import defaultdict
import math

class GaussianBayesClassifier():
    def __init__(self):
        self.train_set = None
        self.train_labels = None
        
    def fit(self, train_set, train_labels):
        self.train_set = train_set
        self.train_labels = train_labels
        self.train_mean_variances = defaultdict(list)  # mean/variances per class
        
        # Loop through each class
        class_set = []
        for i, label in enumerate(train_labels):
            class_set.append(train_set[i])

            if (i == len(train_labels)-1 or label != train_labels[i+1]):  # last element or next label different
                # Calculate mean/variance of class and reset
    
                # Loop through the all pixel indices
                for j in range(len(train_set[0])):
                    # Calculate mean and variance per pixel across the class set
                    pixel_mean = np.mean([img[j] for img in class_set])
                    pixel_variance = np.var([img[j] for img in class_set])

                    self.train_mean_variances[label].append((pixel_mean, pixel_variance))
                class_set = []
    
    def __calc_gaussian_probability(self, x, mean, variance):
        # Gaussian PDF (given mean and variance)
        scalar = (1 / (math.sqrt(2 * math.pi * variance)))
        return scalar * math.exp(-(math.pow(x - mean, 2) / (2 * variance)))
        
    def predict(self, test_image):
        if self.train_set is None or self.train_labels is None:
            raise Exception
        
        # Get per image log likelihood (sum of logs) for every unique class
        likelihoods = {}
        for person_class in set(self.train_labels):
            image_likelihood = 0

            # Get per pixel likelihood for every class
            pixel_likelihoods = []
            for idx_pixel, pixel in enumerate(test_image):
                mean, variance = self.train_mean_variances[person_class][idx_pixel]
                pixel_likelihood = self.__calc_gaussian_probability(pixel, mean, variance)
                if pixel_likelihood > 0:
                    pixel_likelihoods.append(math.log(pixel_likelihood))
            image_likelihood = np.sum(pixel_likelihoods)
            # NOTE: Add prior calculation here if nonuniform distribution
            likelihoods[person_class] = image_likelihood
        
        # Get class with highest likelihood
        max_class = max(likelihoods.keys(), key=(lambda key: likelihoods[key]))
        return max_class
    
    def score(self, test_set, test_labels):
        correct = 0.
        for i in range(len(test_set)):
            if self.predict(test_set[i]) == test_labels[i]:
                correct += 1.
        return correct / len(test_set)
