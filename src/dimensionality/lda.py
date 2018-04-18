import numpy as np
from scipy import linalg 

def lda(train):
    # Get mean vectors for each class (person) of training set
    mean_vectors = []
    for person in train.persons:
        avg = sum([face.vector for face in person.faces])/len(person.faces)
        mean_vectors.append(avg)

    # Compute within-class scatter matrix
    S_W = np.zeros((len(mean_vectors[0]), len(mean_vectors[0])))
    for i, person in enumerate(train.persons):
        S_i = np.zeros((len(mean_vectors[0]), len(mean_vectors[0])))
        for face in person.faces:
            fv = face.vector[np.newaxis].T
            mv = mean_vectors[i][np.newaxis].T
            S_i += (fv - mv) @ (fv - mv).T
        S_W += S_i

    # Compute between-class scatter matrix
    complete_mean = np.mean([face.vector for person in train.persons for face in person.faces], axis=0)  # mean, should be ~0

    S_B = np.zeros((len(mean_vectors[0]), len(mean_vectors[0])))
    for i, _ in enumerate(mean_vectors):
        n = len(train.persons[i].faces)  # redundant because constant class size
        cm = complete_mean[np.newaxis].T
        mv = mean_vectors[i][np.newaxis].T
        S_B += n * (mv - cm) @ (mv - cm).T

    return linalg.eigh(np.linalg.inv(S_W) @ (S_B))