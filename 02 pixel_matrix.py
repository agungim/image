import cv2
import numpy as np
import csv


img = cv2.imread("image/gr_10k.jpg",1)
h, w, c = img.shape
print("Dimensions of the image is:nnHeight:", h, "pixelsnWidth:", w, "pixelsnNumber of Channels:", c)

x1=12
x2=421
y1=330
y2=61

matrix = np.asarray(img)
print('matrix ',matrix)
m=(y2-y1)/(x2-x1)
b=y1-m*x1
print('m ',m,' ',b)
arr=[]
xmax=max(x1,x2)
ymax=max(y1,y2)
for x in range(x1,x2):
    for y in range(y1,y2):
        y3=m*x+b
        print('cek 4 ',int(y3),' ',y)
        #s=cnt+str(matrix[x][y])
        if int(y3)==int(y):
            print('cek')
            arr.append(matrix[x][y])

filename='image/10k.csv'
np.savetxt(filename, arr, delimiter=",")
arr2=[]
arr4=[]
with open(filename) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        str1=''
        str1=str(row[0])
        print(row[0])
        str1=str1+','+str(line_count)
        line_count += 1

        arr2.append(str1)
        arr4.append(row[0])
        '''
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
            line_count += 1
        '''    
    print(f'Processed {line_count} lines.')

filename2='image/10k.txt'
with open(filename2, 'w') as f:
    for line in arr2:
        f.write(line)
        f.write('\n')   


'''
from scipy.fft import fft, fftfreq
from matplotlib import pyplot as plt
# Number of samples in normalized_tone
N = 110
print(arr4)
yf = fft(arr4)
xf = fftfreq(N, 1 / 10)[:N]

plt.plot(xf, np.abs(yf))
plt.show()
'''
