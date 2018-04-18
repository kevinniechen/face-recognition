import numpy as np
import scipy as sp
import scipy.io as sio
import matplotlib.pyplot as plt
from PIL import Image
import random as rand
import pathlib
from shutil import copyfile
import os
import random
from IPython.display import display
import imageio

### IMPORTING AND SAVING IMAGES
DATAPATH = '../data/'

# Create data directories
pathlib.Path(DATAPATH + 'interim/data').mkdir(parents=True, exist_ok=True)
pathlib.Path(DATAPATH + 'interim/illumination').mkdir(parents=True, exist_ok=True)
pathlib.Path(DATAPATH + 'interim/pose').mkdir(parents=True, exist_ok=True)
pathlib.Path(DATAPATH + 'processed/data').mkdir(parents=True, exist_ok=True)
pathlib.Path(DATAPATH + 'processed/illumination').mkdir(parents=True, exist_ok=True)
pathlib.Path(DATAPATH + 'processed/pose').mkdir(parents=True, exist_ok=True)

# Recover image from vector
def recover_img(v, dim1=40, dim2=48, rotate=270):
    rescaled = v.reshape(dim1, dim2)
    return Image.fromarray(rescaled).rotate(rotate, expand=True)

# Load illumination: cropped 40x48 (1920px) images of 68 subjects under 21 different illuminations
illum_dat = sio.loadmat(DATAPATH + '/raw/illumination.mat')
illum_mat = illum_dat['illum']

# Save images as person_67_illum_20.tif
for illum in range(illum_mat.shape[1]):
    for person in range(illum_mat.shape[2]):
        # Create class folder
        person_no, illum_no = '{0:0=2d}'.format(person), '{0:0=2d}'.format(illum)
        pathlib.Path(DATAPATH + 'interim/illumination/person{0}'.format(person_no)).mkdir(parents=True, exist_ok=True)
        # Create and save image
        img = recover_img(illum_mat[:,illum,person])
        filename = 'person{0}_illum{1}.tif'.format(person_no, illum_no)
        img.save(DATAPATH + 'interim/illumination/person{0}/{1}'.format(person_no, filename))

### TRAIN/TEST SPLIT
# Check if image is Tif
def isTif(filename): return filename.endswith(".tif") and '.ipynb_checkpoints' not in filename

# Create test/train split
random.seed(999)
split = 0.8  # 80% of 21 illuminations will be ceilinged to 17, or ~81%
train, test = [], []
subdir = DATAPATH + 'interim/illumination/'
dirs = next(os.walk(subdir))[1]  # class folders

for dir in dirs:
    filenames = os.listdir(subdir + dir)
    img_filenames = [filename for filename in filenames if isTif(filename)]
    img_filepaths = [subdir + dir + os.sep + img_filename for img_filename in img_filenames]
    random.shuffle(img_filepaths) # shuffle class images so train/test have same distribution
    train = train + img_filepaths[:int(split*len(img_filenames))]
    test = test + img_filepaths[int(split*len(img_filenames)):]

### PREPROCESSING
def load_image(filepath):
    return imageio.imread(filepath).flatten(order='F')  # column-major flatten

# Mean center and save images into train/test splits
def center_image(filename):
    v = load_image(filename)
    return v - average_train

def average_image(dataset):
    average_image = [0]
    for filename in dataset:
        average_image += load_image(filename)
    average_image /= len(dataset)
    return average_image

average_train = average_image(train)

pathlib.Path(DATAPATH + 'processed/illumination/train').mkdir(parents=True, exist_ok=True)
pathlib.Path(DATAPATH + 'processed/illumination/test').mkdir(parents=True, exist_ok=True)

for filename in train:
    chunks = filename.split('/')
    # create class folder
    pathlib.Path(DATAPATH + 'processed/illumination/train/' + chunks[-2]).mkdir(parents=True, exist_ok=True)
    # center image and create new file
    new_filepath = '/'.join(["../data/processed/illumination/train"] + chunks[-2:])
    img = recover_img(center_image(filename))
    img.save(new_filepath)
    
for filename in test:
    chunks = filename.split('/')
    # create class folder 
    pathlib.Path(DATAPATH + 'processed/illumination/test/' + chunks[-2]).mkdir(parents=True, exist_ok=True)
    # center image and create new file
    new_filepath = '/'.join(["../data/processed/illumination/test"] + chunks[-2:])
    img = recover_img(center_image(filename))
    img.save(new_filepath)