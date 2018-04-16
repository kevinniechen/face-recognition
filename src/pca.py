""" Principal Component Analysis (PCA)
"""

import numpy as np
import scipy.io as sio

mat_contents = sio.loadmat('../Data/illumination.mat')
print(mat_contents)