# import required libraries
import numpy as gfg
import matplotlib.image as img
import pandas as pd
 
# read an image
imageMat = img.imread('image/agung1.jpg')
print("Image shape:",
      imageMat.shape)
 
# if image is colored (RGB)
if(imageMat.shape[2] == 3):
   
  # reshape it from 3D matrice to 2D matrice
  imageMat_reshape = imageMat.reshape(imageMat.shape[0],
                                      -1)
  print("Reshaping to 2D array:",
        imageMat_reshape.shape)
 
# if image is grayscale
else:
  # remain as it is
  imageMat_reshape = imageMat
     
# converting it to dataframe.
mat_df = pd.DataFrame(imageMat_reshape)
 
# exporting dataframe to CSV file.
mat_df.to_csv('image/agung1.csv',
              header = None,
              index = None)
 
# retrieving dataframe from CSV file
loaded_df = pd.read_csv('image/agung1.csv',
                        sep = ',',
                        header = None)
# getting matrice values.
loaded_2D_mat = loaded_df.values
 
# reshaping it to 3D matrice
loaded_mat = loaded_2D_mat.reshape(loaded_2D_mat.shape[0],
                                   loaded_2D_mat.shape[1] // imageMat.shape[2],
                                   imageMat.shape[2])
 
print("Image shape of loaded Image :",
      loaded_mat.shape)
 
# check if both matrice have same shape or not
if((imageMat == loaded_mat).all()):
  print("\n\nYes",
        "The loaded matrice from CSV file is same as original image matrice")