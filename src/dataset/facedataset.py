from glob import glob
from dataset import person
    
class FaceDataset():
    """Illumination Dataset."""
    
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.persons = [person.Person(dir) for dir in glob(root_dir + '*/')]
        
    @property
    def images(self):
        return glob(self.root_dir + '/*/*.png')
        
    @property
    def num_persons(self):
        return len(self.persons)
    
    @property
    def num_images(self):
        return sum([len(person) for person in self.persons])
        