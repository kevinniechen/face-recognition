from PIL import Image
import imageio

class Face():
    def __init__(self, filepath):
        self.vector = self.load_image(filepath)
    
    @property
    def image(self):
        return self.recover_img(self.vector)
        
    def load_image(self, filepath):
        return imageio.imread(filepath).flatten(order='F')  # column-major flatten

    def recover_img(self, v, dim1=40, dim2=48, rotate=270):
        rescaled = v.reshape(dim1, dim2)
        return Image.fromarray(rescaled).rotate(rotate, expand=True)
    
    def __str__(self):
        return "Face<" + str(self.vector) + ">"