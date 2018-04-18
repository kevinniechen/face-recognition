import numpy as np
from scipy import linalg

def pca(train):
    # Form Image Matrix
    m = np.asarray(train.persons[0].faces[0].vector)[np.newaxis].T
    for person in train.persons:
        for face in person.faces:
            v = face.vector[np.newaxis].T
            m = np.append(m, v, axis=1)

    # Calculate Covariance
    cov = np.cov(m)

    # Calculate eigenvalues/eigenvectors
    eigvals, eigvecs = linalg.eigh(cov)

    # Sort by top eigenvalues
    idx = eigvals.argsort()[::-1]   
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:,idx]

    return (eigvals, eigvecs)