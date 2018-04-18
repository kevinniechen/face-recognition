from dataset import face
from glob import glob
import os

class Person():
    """Person (for a series of face photos)."""
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.faces = [face.Face(file) for file in self.get_image_files(root_dir)]
        
    @property
    def id(self):
        person_str = self.root_dir.split('/')[-2]
        person_no = int(''.join([s for s in person_str if s.isdigit()]))
        return person_no
        
    def get_image_files(self, dir):
        return [filename for filename in glob(dir + os.sep + '*.tif')]
    
    def __str__(self):
        return "Person<" + str(self.id) + ">"