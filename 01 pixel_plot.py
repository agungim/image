import numpy as np
from PIL import Image

# load and display an image with Matplotlib
from matplotlib import image
from matplotlib import pyplot
# load image as pixel array

filename='image/10k.jpg'
filename2='image/gr_10k.jpg'
image = image.imread(filename)
# summarize shape of the pixel array
print(image.dtype)
print(image.shape)
# display the array of pixels as an image
pyplot.imshow(image)
pyplot.show()



im = np.array(Image.open(filename).convert('L')) #you can pass multiple arguments in single line
print(type(im))

gr_im= Image.fromarray(im).save(filename2)