import numpy as np
from PIL import Image

# load and display an image with Matplotlib
from matplotlib import image
from matplotlib import pyplot
# load image as pixel array
image = image.imread('image/1k.jpg')
# summarize shape of the pixel array
print(image.dtype)
print(image.shape)
# display the array of pixels as an image
pyplot.imshow(image)
pyplot.show()



im = np.array(Image.open('image/1k.jpg').convert('L')) #you can pass multiple arguments in single line
print(type(im))

gr_im= Image.fromarray(im).save('image/gr_1k.jpg')