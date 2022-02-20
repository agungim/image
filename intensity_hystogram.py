import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("image/night.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Grayscale Moonlight Mountains", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


import matplotlib.pyplot as plt
plt.hist(x=image.ravel(), bins=256, range=[0, 256], color='crimson')
plt.title("Histogram Showing Pixel Intensities And Counts", color='crimson')
plt.ylabel("Number Of Pixels Belonging To The Pixel Intensity", color="crimson")
plt.xlabel("Pixel Intensity", color="crimson")
plt.show()

image_enhanced = cv2.equalizeHist(src=image)
cv2.imshow("Enhanced Contrast", image_enhanced)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.hist(image_enhanced.ravel(), 256, [0,256], color="blue")
plt.title("Pixel Intensities And Counts In Enhanced Image", color="crimson")
plt.ylabel("Number Of Pixels Belonging To Pixel Intensity", color="crimson")
plt.xlabel("Pixel Intensity", color="crimson")
plt.show()