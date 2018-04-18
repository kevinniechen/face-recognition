from queue import PriorityQueue
import math
from collections import defaultdict

class KNNClassifier():
    def __init__(self, neighbors=1):
        self.neighbors = neighbors
        self.train_set = None
        self.train_labels = None
        
    def fit(self, train_set, train_labels):
        self.train_set = train_set
        self.train_labels = train_labels

    def __euclideanDistance(self, v1, v2):
        if (len(v1) != len(v2)):
            raise Exception("Vectors not equal")
        distance = 0
        for x in range(min(len(v1),len(v2))):
            distance += math.pow((v1[x] - v2[x]), 2)
        return math.sqrt(distance)

    def __getNeighbors(self, train_set, train_labels, test, k):
        distances = PriorityQueue()
        for i in range(len(train_set)):
            distances.put((self.__euclideanDistance(test, train_set[i]), train_labels[i]))
        neighbors = []
        for _ in range(k):
            neighbors.append(distances.get())
        return neighbors

    def __getMajorityVote(self, neighbors):
        votes = defaultdict(int)
        for i in range(len(neighbors)):
            label = neighbors[i][-1]
            votes[label] += 1
        return sorted(votes.items(), key=lambda k_v: k_v[1])[-1][0]
    
    def predict(self, test_image):
        if self.train_set is None or self.train_labels is None:
            raise Exception
            
        neighbors = self.__getNeighbors(self.train_set, self.train_labels, test_image, self.neighbors)
        response = self.__getMajorityVote(neighbors)
        return response
    
    def score(self, test_set, test_labels):
        correct = 0.
        for i in range(len(test_set)):
            if self.predict(test_set[i]) == test_labels[i]:
                correct += 1.
        return correct / len(test_set)