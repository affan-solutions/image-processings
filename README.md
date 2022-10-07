# image-processings
image segmentation , object detections, 



# Imports

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, io, segmentation, util, measure, morphology
from copy import deepcopy

plt.rcParams['figure.dpi'] = 200

# Question 1.

Find the vertical and horizontal edges in the following figure.
<br><br><img height=128 width=196 src="./Figure 8.jpg" alt="Figure 8">

# Read Image
book_im = io.imread('./Figure 8.jpg', as_gray=True)
plt.imshow(book_im)





class EdgeDetection:
    def __init__(self, image) -> None:
        self._filters_dict = {
            'sobel': [filters.sobel_h, filters.sobel_v],
            'prewitt': [filters.prewitt_h, filters.prewitt_v],
            'farid': [filters.farid_h, filters.farid_v],
            'scharr': [filters.scharr_h, filters.scharr_v],
        }
        self._image = image

    def getFilter(self, name):
        return self._filters_dict[name]

    def detect(self, filter):
        ftype = self.getFilter(filter)
        
        return [filter]+[f(self._image) for f in ftype]

    def plot(self, data):
        # Create two subplots and unpack the output array immediately
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
        ax1.imshow(self._image)
        ax1.set_title("Original Image")
        ax2.imshow(data[1])
        ax2.set_title(data[0].title() + ' Horizontal')
        ax3.imshow(data[2])
        ax3.set_title(data[0].title() + ' Vertical')
        
        plt.show()

ed = EdgeDetection(book_im)
# Using Sobel Filter
ed.plot(ed.detect('sobel'))
# Using Prewitt Filter
ed.plot(ed.detect('prewitt'))
# Using Farid Transform
ed.plot(ed.detect('farid'))
# Using Scharr Transform
ed.plot(ed.detect('scharr'))


# Question 2.

Crop the image to get another image that contains window. Use correlation, to find all the windows in Figure 2.jpg and draw a bounding box. Hint: Crop the window from the image and use it as a template to find all windows in original image.
<br><br><img height=128 width=196 src="./Figure 2.jpg" alt="Figure 2">

house_rgb = cv2.imread('./Figure 2.jpg')
house = cv2.cvtColor(house_rgb, cv2.COLOR_BGR2GRAY)
plt.imshow(house)

blurred_house = cv2.GaussianBlur(house,(5,5),0)
laplacian = cv2.Laplacian(blurred_house, cv2.CV_32F)
plt.imshow(laplacian)

template = cv2.imread('./fig2_template.jpg', 0)
template = template[1: -1, 1: -2]
plt.figure(figsize=(3,2))
plt.imshow(template)

blurred_template = cv2.GaussianBlur(template,(3,3),0)
laplacian_template = cv2.Laplacian(blurred_template, cv2.CV_32F)
plt.figure(figsize=(1.5,0.75))
plt.imshow(laplacian_template)

height, width = template.shape[::]

res = cv2.matchTemplate(laplacian, laplacian_template, cv2.TM_CCOEFF_NORMED)
plt.imshow(res, cmap='gray')

threshold = 0.5 #For TM_CCOEFF_NORMED, larger values = good fit.

loc = np.where( res >= threshold)

resultant = deepcopy(house_rgb)
for pt in zip(*loc[::-1]): 
    cv2.rectangle(resultant, pt, (pt[0] + width, pt[1] + height), (255, 0, 0), 1) 

plt.imshow(resultant)

# Question 3.

Segment the cloth in Figure 4.jpg
<br><br><img height=256 width=256 src="./Figure 4.jpg" alt="Figure 4">

# Read Image
cloth_im = cv2.imread('Figure 4.jpg')
# convert the image to grayscale and blur it slightly
cloth_im_gray = cv2.cvtColor(cloth_im, cv2.COLOR_BGR2GRAY)
blurred_cloth = cv2.GaussianBlur(cloth_im_gray, (11, 11), 0)
plt.imshow(blurred_cloth)

histr = cv2.calcHist([blurred_cloth],[0],None,[256],[0,256])
  
# show the plotting graph of an image
plt.figure(figsize=(6,3))
plt.plot(histr)
plt.show()

ret, thresh = cv2.threshold(blurred_cloth, 135,  0, cv2.THRESH_TOZERO_INV)
plt.imshow(thresh)

thresh = cv2.adaptiveThreshold(blurred_cloth, 255,
	cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 47, 18)

plt.imshow(thresh)

resultant = morphology.remove_small_objects(util.img_as_bool(thresh), 90, connectivity=2)
plt.imshow(resultant)

resultantB = morphology.remove_small_objects(resultant, 196, connectivity=2)
plt.imshow(resultantB)

# Question 4.

You are provided with an image labelled ‘disk’, design an algorithm to find its corner points and highlight these points on the original image using circles of radius 5.
<br><br><img height=128 width=196 src="./disk.jpg" alt="Disk">

disk_rgb = cv2.imread('./disk.jpg')
disk = cv2.cvtColor(disk_rgb, cv2.COLOR_BGR2GRAY)
plt.imshow(disk)

gray = np.float32(disk)
dst = cv2.cornerHarris(gray,blockSize=2,ksize=3,k=0.1)
print('Corner Points: ', np.count_nonzero(dst>0.05*dst.max()))

cloc = np.where( dst>0.05*dst.max() )

resultant = deepcopy(disk_rgb)
for pt in zip(*cloc[::-1]): 
    cv2.circle(resultant, (pt[0], pt[1]), 5, (255, 0, 0), 1)

plt.imshow(resultant)

