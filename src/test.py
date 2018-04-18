""" Test fixed classifiers

written by Kevin Chen"""

import numpy as np
import os

from classifiers.knn import KNNClassifier
from classifiers.bayes import GaussianBayesClassifier
from dataset import facedataset
from dimensionality.pca import pca
from dimensionality.lda import lda

# os.chdir('/Users/kev/Code/classes/cmsc498m/Project1/src')


def project_eigenface(M, vectors):
    return [M @ v for v in vectors]


########################################################
# TRAIN TEST
########################################################
print("Splitting train/test (train: 80, test:20)...")

TRAINPATH = '../data/processed/illumination/train/'
TESTPATH = '../data/processed/illumination/test/'

train = facedataset.FaceDataset(TRAINPATH)
test = facedataset.FaceDataset(TESTPATH)

# Get image vectors and class labels of training set
train_X = []
train_Y = []
for person in train.persons:
    for face in person.faces:
        train_X.append(face.vector)
        train_Y.append(person.id)

# Get image vectors and class labels of test set
test_X = []
test_Y = []
for person in test.persons:
    for face in person.faces:
        test_X.append(face.vector)
        test_Y.append(person.id)


########################################################
# PCA
########################################################
print("PCA...")
p_eigvals, p_eigvecs = pca(train)


########################################################
# LDA
########################################################
print("LDA...")
l_eigvals, l_eigvecs = lda(train)


########################################################
# KNN WITH PCA
########################################################
print("KNN with PCA... (k=1, eigvecs=30, deleted_eigvecs=5)")
pr_train_X = project_eigenface(p_eigvecs[:,5:30].T, train_X)
pr_test_X = project_eigenface(p_eigvecs[:,5:30].T, test_X)
model = KNNClassifier(neighbors=1)
model.fit(pr_train_X, train_Y)
acc = model.score(pr_test_X, test_Y)
print("Accuracy: " + str(round(acc, 3)))


########################################################
# KNN WITH LDA
########################################################
print("KNN with LDA... (k=1, eigvecs=30, deleted_eigvecs=5")
pr_train_X = project_eigenface(l_eigvecs[:,5:30].T, train_X)
pr_test_X = project_eigenface(l_eigvecs[:,5:30].T, test_X)
model = KNNClassifier(neighbors=1)
model.fit(pr_train_X, train_Y)
acc = model.score(pr_test_X, test_Y)
print("Accuracy: " + str(round(acc, 3)))


########################################################
# BAYES WITH PCA
########################################################
print("BAYES with PCA... (k=1, eigvecs=50, deleted_eigvecs=0)")
pr_train_X = project_eigenface(p_eigvecs[:,:50].T, train_X)
pr_test_X = project_eigenface(p_eigvecs[:,:50].T, test_X)
model = GaussianBayesClassifier()
model.fit(pr_train_X, train_Y)
acc = model.score(pr_test_X, test_Y)
print("Accuracy: " + str(round(acc, 3)))


########################################################
# BAYES WITH LDA
########################################################
print("BAYES with LDA... (k=1, eigvecs=50, deleted_eigvecs=0)")
pr_train_X = project_eigenface(l_eigvecs[:,:50].T, train_X)
pr_test_X = project_eigenface(l_eigvecs[:,:50].T, test_X)
model = GaussianBayesClassifier()
model.fit(pr_train_X, train_Y)
acc = model.score(pr_test_X, test_Y)
print("Accuracy: " + str(round(acc, 3)))


########################################################
# ENDING MESSAGE
########################################################
print("Run the Jupyter Notebooks for Plots and Exploratory Analysis!")