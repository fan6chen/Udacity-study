import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import time
from skimage.feature import hog
from scipy.ndimage.measurements import label
from functions import *

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, scale):
    img_boxes = []
    draw_img = np.copy(img)
    heatmap = np.zeros_like(img[:,:,0])
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,xstart:xstop,:]
    ctrans_tosearch = convert_color(img_tosearch, conv)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - 1 
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - 1 
    cells_per_step = 1  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))      
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left+xstart, ytop_draw+ystart),(xbox_left+win_draw+xstart,ytop_draw+win_draw+ystart),(0,0,255),3) 
                img_boxes.append(((xbox_left+xstart, ytop_draw+ystart), (xbox_left+win_draw+xstart,ytop_draw+win_draw+ystart)))
                heatmap[ytop_draw+ystart:ytop_draw+win_draw+ystart, xbox_left+xstart:xbox_left+win_draw+xstart] += 1
    
    return img_boxes, heatmap

    
def process_image(img):
    global process_num
    global carlist
    process_num += 1
    #print('Image Num:', process_num)
    # use different scale to find cars
    scale = 1.5
    img_boxes1, heat_map1 = find_cars(img, scale)
    scale = 1
    img_boxes2, heat_map2 = find_cars(img, scale)
    scale = 2
    img_boxes3, heat_map3 = find_cars(img, scale)
    for i in range(len(img_boxes2)):
        img_boxes1.append(img_boxes2[i])
    for i in range(len(img_boxes3)):
        img_boxes1.append(img_boxes3[i])

    draw_img = draw_boxes(img, img_boxes1)
    #cv2.imwrite('./output_images/draw_img'+str(process_num)+'.jpg', draw_img)

    heatmap = np.zeros_like(img[:,:,0])
    heatmap = add_heat(heatmap, img_boxes1)
    heatmap = apply_threshold(heatmap, 3)
    #cv2.imwrite('./output_images/heatmap_img'+str(process_num)+'.jpg', heatmap)
    labels = label(heatmap)

    final_draw_img = np.copy(img)
    car_found = []

    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        xmin = np.min(nonzerox)
        xmax = np.max(nonzerox)
        ymin = np.min(nonzeroy)
        ymax = np.max(nonzeroy)
        x_position = (xmin+xmax)/2
        y_position = (ymin+ymax)/2
        exist = False

        for car in carlist:
            if x_position < car.xpixels+120 and x_position > car.xpixels-120 and y_position < car.ypixels+120 and y_position > car.ypixels-120:
                exist = True
                car_found.append(carlist.index(car))
                car.n_detections += 1
                car.xpixels = x_position
                car.ypixels = y_position
                car.recent_xfitted.append(car.xpixels)
                car.recent_yfitted.append(car.ypixels)
                car.recent_wfitted.append(xmax-xmin)
                car.recent_hfitted.append(ymax-ymin)
                #if car.detected == True:
                car.bestx = np.average(car.recent_xfitted)
                car.besty = np.average(car.recent_yfitted)
                car.bestw = np.average(car.recent_wfitted)
                car.besth = np.average(car.recent_hfitted)
                #else:
                #    car.bestx = car.xpixels
                #    car.besty = car.ypixels
                #    car.bestw = xmax - xmin
                #    car.besth = ymax - ymin
                car.detected = True
                bbox = ((int(car.xpixels-car.bestw/2), int(car.besty-car.besth/2)), (int(car.xpixels+car.bestw/2), int(car.besty+car.besth/2)))
                cv2.rectangle(final_draw_img, bbox[0], bbox[1], (0,0,255), 3)
                #print('Exist Car Index:', carlist.index(car), 'x:', round(car.bestx, 0), ' y:', round(car.besty, 0),'Width:', round(car.bestw, 0))
            if exist == True:
                break
        if exist == False:
            car = Vehicle()
            car.n_detections += 1
            car.n_nondetections = process_num - 1
            car.xpixels = x_position
            car.ypixels = y_position
            car.recent_xfitted.append(car.xpixels)
            car.recent_yfitted.append(car.ypixels)
            car.recent_wfitted.append(xmax-xmin)
            car.recent_hfitted.append(ymax-ymin)
            car.bestx = np.average(car.recent_xfitted)
            car.besty = np.average(car.recent_yfitted)
            car.bestw = np.average(car.recent_wfitted)
            car.besth = np.average(car.recent_hfitted)
            car.detected = True
            carlist.append(car)
            bbox = ((xmin, ymin), (xmax, ymax))
            cv2.rectangle(final_draw_img, bbox[0], bbox[1], (0,0,255), 3)
            print('New Car Index:', carlist.index(car), 'x:', round(car.bestx, 0), ' y:', round(car.besty, 0),'Width:', round(car.bestw, 0))
    for car in carlist:
        if carlist.index(car) in car_found:
            continue
        else:
            car.detected = False
    #draw_img = draw_labeled_bboxes(np.copy(img), labels)
    #cv2.imwrite('./output_images/finaldraw_img'+str(process_num)+'.jpg', final_draw_img)
    return final_draw_img

dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]

# Consider a narrower swath in y
ystart = 400
ystop = 656
xstart = 640
xstop = 1280
conv = 'YCrCb'
process_num = 0
carlist = []

# Make a list of the test images
#images = glob.glob('./test_images/test*.jpg')

#for idx, fname in enumerate(images):
#    img = mpimg.imread(fname)
#    draw_img = process_image(img)
#    write_name = './test_images/detect_img'+str(idx+1)+'.jpg'
#    cv2.imwrite(write_name, draw_img)

