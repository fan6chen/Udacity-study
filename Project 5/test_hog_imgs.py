import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
import glob
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from functions import *

# Read in and make lists of cars and notcars
cars_images = glob.glob('/home/clement/CarND-Vehicle-Detection/vehicles/*.jpg')
notcars_images = glob.glob('/home/clement/CarND-Vehicle-Detection/non-vehicles/*.jpg')

# Read in cars and notcars
cars = []
notcars = []
for image in cars_images:
    cars.append(image)
for image in notcars_images:
    notcars.append(image)

random_idxs = np.random.randint(0, len(cars))
car_image = mpimg.imread(cars[random_idxs])
notcar_image = mpimg.imread(notcars[random_idxs])

color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 6  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

car_features, car_hog_image = single_img_features(car_image, color_space=color_space, spatial_size=spatial_size,
                        hist_bins=hist_bins, orient=orient,
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat, vis=True)
notcar_features, notcar_hog_image = single_img_features(notcar_image, color_space=color_space, spatial_size=spatial_size,
                        hist_bins=hist_bins, orient=orient,
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat, vis=True)

write_name = './output_images/car_image.jpg'
cv2.imwrite(write_name, car_image)
cv2.imwrite('./output_images/car_hog_image.jpg', car_hog_image)
write_name = './output_images/notcar_image.jpg'
cv2.imwrite(write_name, notcar_image)
cv2.imwrite('./output_images/notcar_hog_image.jpg', notcar_hog_image)