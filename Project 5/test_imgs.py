import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
import glob
import time
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
from functions import *

dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]

images = glob.glob('./test_images/*.jpg')

color_space = 'YCrCb'
y_start_stop = [400, 656]
overlap = 0.5
hog_channel = 'ALL'
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
n = 0

for img_src in images:
	n += 1
	t1 = time.time()
	img = mpimg.imread(img_src)
	draw_img = np.copy(img)
	img = img.astype(np.float32)/255
	print(np.min(img), np.max(img))

	windows = slide_window(img, x_start_stop=[None, None], y_start_stop=y_start_stop,
		               xy_window=(64, 64), xy_overlap = (overlap, overlap))
	hot_windows = search_windows(img, windows, svc, X_scaler, color_space=color_space,
		                    spatial_size=spatial_size, hist_bins=hist_bins,
		                    orient=orient, pix_per_cell=pix_per_cell,
		                    cell_per_block=cell_per_block,
		                    hog_channel=hog_channel, spatial_feat=spatial_feat,
		                    hist_feat=hist_feat, hog_feat=hog_feat)
	window_img = draw_boxes(draw_img, hot_windows, color=(0, 0, 255), thick=6)
	cv2.imwrite('./test_images/result'+str(n)+'.jpg', window_img)
	print(time.time()-t1, 'seconds to process one image searching', len(windows), 'windows')