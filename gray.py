import numpy as np
import skimage.color
import skimage.io
import matplotlib.pyplot as plt
#%matplotlib widget

# read the image of a plant seedling as grayscale from the outset
image = skimage.io.imread(fname='image/agung1.jpg', as_gray=True)

# display the image
fig, ax = plt.subplots()
plt.imshow(image, cmap='gray')
plt.show()

histogram, bin_edges = np.histogram(image, bins=256, range=(0, 1))
# configure and draw the histogram figure
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("grayscale value")
plt.ylabel("pixel count")
plt.xlim([0.0, 1.0])  # <- named arguments do not work here

plt.plot(bin_edges[0:-1], histogram)  # <- or here
plt.show()