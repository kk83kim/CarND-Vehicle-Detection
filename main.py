import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from feature_functions import *
import pickle
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
from collections import deque


# # Extract features from a single image window
# # This function is very similar to extract_features()
# # just for a single image rather than list of images
# def single_img_features(img, color_space='BGR', spatial_size=(32, 32),
#                         hist_bins=32, orient=9,
#                         pix_per_cell=8, cell_per_block=2, hog_channel=0,
#                         spatial_feat=True, hist_feat=True, hog_feat=True):
#     # 1) Define an empty list to receive features
#     img_features = []
#     # 2) Apply color conversion if other than 'BGR'
#     if color_space != 'BGR':
#         if color_space == 'HSV':
#             feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#         elif color_space == 'LUV':
#             feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
#         elif color_space == 'HLS':
#             feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
#         elif color_space == 'YUV':
#             feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
#         elif color_space == 'YCrCb':
#             feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
#     else:
#         feature_image = np.copy(img)
#     # 3) Compute spatial features if flag is set
#     if spatial_feat == True:
#         spatial_features = bin_spatial(feature_image, size=spatial_size)
#         # 4) Append features to list
#         img_features.append(spatial_features)
#     # 5) Compute histogram features if flag is set
#     if hist_feat == True:
#         hist_features = color_hist(feature_image, nbins=hist_bins)
#         # 6) Append features to list
#         img_features.append(hist_features)
#     # 7) Compute HOG features if flag is set
#     if hog_feat == True:
#         if hog_channel == 'ALL':
#             hog_features = []
#             for channel in range(feature_image.shape[2]):
#                 hog_features.extend(get_hog_features(feature_image[:, :, channel],
#                                                      orient, pix_per_cell, cell_per_block,
#                                                      vis=False, feature_vec=True))
#         else:
#             hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
#                                             pix_per_cell, cell_per_block, vis=False, feature_vec=True)
#         # 8) Append features to list
#         img_features.append(hog_features)
#
#     # 9) Return concatenated array of features
#     return np.concatenate(img_features)
#
#
# # Find list of windows containing cars
# def search_windows(img, windows, clf, scaler, color_space='BGR',
#                    spatial_size=(32, 32), hist_bins=32,
#                    hist_range=(0, 256), orient=9,
#                    pix_per_cell=8, cell_per_block=2,
#                    hog_channel=0, spatial_feat=True,
#                    hist_feat=True, hog_feat=True):
#     # 1) Create an empty list to receive positive detection windows
#     on_windows = []
#     # 2) Iterate over all windows in the list
#     for window in windows:
#         # 3) Extract the test window from original image
#         test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
#         # 4) Extract features for that window using single_img_features()
#         features = single_img_features(test_img, color_space=color_space,
#                                        spatial_size=spatial_size, hist_bins=hist_bins,
#                                        orient=orient, pix_per_cell=pix_per_cell,
#                                        cell_per_block=cell_per_block,
#                                        hog_channel=hog_channel, spatial_feat=spatial_feat,
#                                        hist_feat=hist_feat, hog_feat=hog_feat)
#         # 5) Scale extracted features to be fed to classifier
#         test_features = scaler.transform(np.array(features).reshape(1, -1))
#         # 6) Predict using your classifier
#         prediction = clf.predict(test_features)
#         # 7) If positive (prediction == 1) then save the window
#         if prediction == 1:
#             on_windows.append(window)
#     # 8) Return windows for positive detections
#     return on_windows
#
# # Function that takes an image,
# # start and stop positions in both x and y,
# # window size (x and y dimensions),
# # and overlap fraction (for both x and y)
# def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
#                  xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
#     # If x and/or y start/stop positions not defined, set to image size
#     if x_start_stop[0] == None:
#         x_start_stop[0] = 0
#     if x_start_stop[1] == None:
#         x_start_stop[1] = img.shape[1]
#     if y_start_stop[0] == None:
#         y_start_stop[0] = 0
#     if y_start_stop[1] == None:
#         y_start_stop[1] = img.shape[0]
#     # Compute the span of the region to be searched
#     xspan = x_start_stop[1] - x_start_stop[0]
#     yspan = y_start_stop[1] - y_start_stop[0]
#     # Compute the number of pixels per step in x/y
#     nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
#     ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
#     # Compute the number of windows in x/y
#     nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
#     ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
#     nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
#     ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
#     # Initialize a list to append window positions to
#     window_list = []
#     # Loop through finding x and y window positions
#     for ys in range(ny_windows):
#         for xs in range(nx_windows):
#             # Calculate window position
#             startx = xs * nx_pix_per_step + x_start_stop[0]
#             endx = startx + xy_window[0]
#             starty = ys * ny_pix_per_step + y_start_stop[0]
#             endy = starty + xy_window[1]
#
#             # Append window position to list
#             window_list.append(((startx, starty), (endx, endy)))
#     # Return the list of windows
#     return window_list


def convert_color(img, conv='BGR2YCrCb'):
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'BGR2LUV':
        return cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    if conv == 'BGR2YUV':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

# Draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    imcopy = np.dstack([imcopy[:, :, 2], imcopy[:, :, 1], imcopy[:, :, 0]])
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    draw_img = np.copy(img)
    img_tosearch = img[ystart:ystop, :, :]

    ctrans_tosearch = convert_color(img_tosearch, conv='BGR2YUV')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    bboxes = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                              (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)
                bboxes.append(((int(xbox_left), int(ytop_draw + ystart)),
                               (int(xbox_left + win_draw), int(ytop_draw + win_draw + ystart))))
    return draw_img, bboxes


def apply_sliding_window(image, svc, X_scaler, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    bboxes = []
    # ystart = 400
    # ystop = 500
    # out_img, bboxes1 = find_cars(image, ystart, ystop, 1.0, svc, X_scaler, orient, pix_per_cell, cell_per_block,
    #                             spatial_size, hist_bins)
    # ystart = 400
    # ystop = 550
    # out_img, bboxes2 = find_cars(out_img, ystart, ystop, 1.5, svc, X_scaler, orient, pix_per_cell, cell_per_block,
    #                             spatial_size, hist_bins)
    # ystart = 400
    # ystop = 600
    # out_img, bboxes3 = find_cars(out_img, ystart, ystop, 2.0, svc, X_scaler, orient, pix_per_cell, cell_per_block,
    #                              spatial_size, hist_bins)
    # ystart = 425
    # ystop = 650
    # out_img, bboxes4 = find_cars(out_img, ystart, ystop, 2.5, svc, X_scaler, orient, pix_per_cell, cell_per_block,
    #                              spatial_size, hist_bins)
    # ystart = 450
    # ystop = 675
    # out_img, bboxes5 = find_cars(out_img, ystart, ystop, 3.0, svc, X_scaler, orient, pix_per_cell, cell_per_block,
    #                              spatial_size, hist_bins)
    # bboxes.extend(bboxes1)
    # bboxes.extend(bboxes2)
    # bboxes.extend(bboxes3)
    # bboxes.extend(bboxes4)
    # bboxes.extend(bboxes5)

    ystart = 400
    ystop = 500
    out_img, bboxes1 = find_cars(image, ystart, ystop, 1.0, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                                 spatial_size, hist_bins)
    ystart = 400
    ystop = 500
    out_img, bboxes2 = find_cars(image, ystart, ystop, 1.3, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                                 spatial_size, hist_bins)
    ystart = 410
    ystop = 500
    out_img, bboxes3 = find_cars(out_img, ystart, ystop, 1.4, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                                 spatial_size, hist_bins)
    ystart = 420
    ystop = 556
    out_img, bboxes4 = find_cars(out_img, ystart, ystop, 1.6, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                                 spatial_size, hist_bins)
    ystart = 430
    ystop = 556
    out_img, bboxes5 = find_cars(out_img, ystart, ystop, 1.8, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                                 spatial_size, hist_bins)
    ystart = 430
    ystop = 556
    out_img, bboxes6 = find_cars(out_img, ystart, ystop, 2.0, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                                 spatial_size, hist_bins)
    ystart = 440
    ystop = 556
    out_img, bboxes7 = find_cars(out_img, ystart, ystop, 1.9, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                                 spatial_size, hist_bins)
    ystart = 400
    ystop = 556
    out_img, bboxes8 = find_cars(out_img, ystart, ystop, 1.3, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                                 spatial_size, hist_bins)
    ystart = 400
    ystop = 556
    out_img, bboxes9 = find_cars(out_img, ystart, ystop, 2.2, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                                 spatial_size, hist_bins)
    ystart = 500
    ystop = 656
    out_img, bboxes10 = find_cars(out_img, ystart, ystop, 3.0, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                                  spatial_size, hist_bins)
    bboxes.extend(bboxes1)
    bboxes.extend(bboxes2)
    bboxes.extend(bboxes3)
    bboxes.extend(bboxes4)
    bboxes.extend(bboxes5)
    bboxes.extend(bboxes6)
    bboxes.extend(bboxes7)
    bboxes.extend(bboxes8)
    bboxes.extend(bboxes9)
    bboxes.extend(bboxes10)


    return out_img, bboxes


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap







model = pickle.load( open("model.p", "rb" ) )
# get attributes of our svc object
clf = model["clf"]
X_scaler = model["scaler"]
orient = model["orient"]                    # HOG orientations
pix_per_cell = model["pix_per_cell"]        # HOG pixels per cell
cell_per_block = model["cell_per_block"]    # HOG cells per block
spatial_size = model["spatial_size"]        # Spatial binning dimensions
hist_bins = model["hist_bins"]              # Number of histogram bins
color_space = 'YUV'                         # BGR, HSV, LUV, HLS, YUV, YCrCb
hog_channel = 'ALL'                         # 0, 1, 2, or "ALL"
spatial_feat = True                         # Spatial features on or off
hist_feat = True                            # Histogram features on or off
hog_feat = True                             # HOG features on or off
y_start_stop = [400, None]                  # Min and max in y to search in slide_window()

# image = cv2.imread('./test_images/test6.jpg')
# draw_image = np.copy(image)

# windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
#                        xy_window=(96, 96), xy_overlap=(0.75, 0.75))
#
# hot_windows = search_windows(image, windows, clf, X_scaler, color_space=color_space,
#                              spatial_size=spatial_size, hist_bins=hist_bins,
#                              orient=orient, pix_per_cell=pix_per_cell,
#                              cell_per_block=cell_per_block,
#                              hog_channel=hog_channel, spatial_feat=spatial_feat,
#                              hist_feat=hist_feat, hog_feat=hog_feat)
#
# window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
#
#
# ystart = 400
# ystop = 656
# scale = 1
# # window_img, bboxes = find_cars(image, ystart, ystop, scale, clf, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
# window_img, bboxes = apply_sliding_window(image, clf, X_scaler, pix_per_cell, cell_per_block, spatial_size, hist_bins)
# window_img = np.dstack([window_img[:, :, 2], window_img[:, :, 1], window_img[:, :, 0]])
# # plt.imshow(window_img)
# # plt.savefig("test.png")
# # plt.show()
#
#
# heat = np.zeros_like(image[:, :, 0]).astype(np.float)
# # Add heat to each box in box list
# heat = add_heat(heat, bboxes)
#
# # Apply threshold to help remove false positives
# threshold = 2
# heat = apply_threshold(heat, threshold)
#
# # Visualize the heatmap when displaying
# heatmap = np.clip(heat, 0, 255)
#
# # Find final boxes from heatmap using label function
# labels = label(heatmap)
# draw_img = draw_labeled_bboxes(np.copy(image), labels)

# print(labels)
# plt.imshow(labels)
# plt.show()
#
# plt.imshow(labels[0], cmap='gray')
#
# plt.imshow(draw_img)
# # plt.savefig("heat_map.png")
# plt.show()





i = 0
def process_image(img):
    img = np.dstack([img[:, :, 2], img[:, :, 1], img[:, :, 0]])
    window_img, bboxes = apply_sliding_window(img, clf, X_scaler, pix_per_cell, cell_per_block, spatial_size, hist_bins)

    # heat map
    threshold_cur = 1
    heat = np.zeros_like(img[:, :, 0]).astype(np.float)
    heat = add_heat(heat, bboxes)
    heat = apply_threshold(heat, threshold_cur)
    heatmap_cur = np.clip(heat, 0, 255)

    heatmap_history.append(heatmap_cur)
    heatmap = np.zeros_like(heatmap_cur).astype(np.float)
    for heat in heatmap_history:
        heatmap = heatmap + heat

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    out_img = draw_labeled_bboxes(np.copy(img), labels)

    window_img = np.dstack([window_img[:, :, 2], window_img[:, :, 1], window_img[:, :, 0]])
    out_img = np.dstack([out_img[:, :, 2], out_img[:, :, 1], out_img[:, :, 0]])

    # plt.imshow(draw_img)
    # plt.savefig("heat_map.png")
    global i
    # cv2.imwrite("./window_imgs/{}.png".format(i), window_img)
    # window_img = np.dstack([window_img[:, :, 2], window_img[:, :, 1], window_img[:, :, 0]])
    # cv2.imwrite("./output_images/test.png", window_img)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(window_img)
    ax2.imshow(heatmap_cur, cmap='hot')
    plt.savefig("./output_images/heatmap{}.png".format(i))

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(labels[0], cmap='gray')
    ax2.imshow(out_img)
    plt.savefig("./output_images/label{}.png".format(i))

    i += 1

    return out_img

heatmap_history = deque(maxlen = 5)
output = 'OUT.mp4'
clip = VideoFileClip("project_video.mp4").subclip(38,39)
clip = clip.fl_image(process_image)
clip.write_videofile(output, audio=False)







# image = cv2.imread('./test_images/test6.jpg')
# process_image(image)

# car_image = cv2.imread('./output_images/car.png')
# notcar_image = cv2.imread('./output_images/not_car.png')
#
# color_space = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
# orient = 15  # HOG orientations
# pix_per_cell = 8 # HOG pixels per cell
# cell_per_block = 2 # HOG cells per block
# hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
# spatial_size = (32, 32) # Spatial binning dimensions
# hist_bins = 32    # Number of histogram bins
# spatial_feat = True # Spatial features on or off
# hist_feat = True # Histogram features on or off
# hog_feat = True # HOG features on or off
#
# converted_car_image = cv2.cvtColor(car_image, cv2.COLOR_BGR2YUV)
# car_ch1 = converted_car_image[:,:,0]
# car_ch2 = converted_car_image[:,:,1]
# car_ch3 = converted_car_image[:,:,2]
#
# converted_notcar_image = cv2.cvtColor(notcar_image, cv2.COLOR_BGR2YUV)
# notcar_ch1 = converted_notcar_image[:,:,0]
# notcar_ch2 = converted_notcar_image[:,:,1]
# notcar_ch3 = converted_notcar_image[:,:,2]
#
# car_hog_feature, car_hog_image = get_hog_features(car_ch1,
#                                         orient, pix_per_cell, cell_per_block,
#                                         vis=True, feature_vec=True)
#
# notcar_hog_feature, notcar_hog_image = get_hog_features(notcar_ch1,
#                                         orient, pix_per_cell, cell_per_block,
#                                         vis=True, feature_vec=True)
#
# car_ch1_features = cv2.resize(car_ch1, spatial_size)
# car_ch2_features = cv2.resize(car_ch2, spatial_size)
# car_ch3_features = cv2.resize(car_ch3, spatial_size)
# notcar_ch1_features = cv2.resize(notcar_ch1, spatial_size)
# notcar_ch2_features = cv2.resize(notcar_ch2, spatial_size)
# notcar_ch3_features = cv2.resize(notcar_ch3, spatial_size)
#
# def show_images(image1, image2, image3, image4,  image1_exp="Image 1", image2_exp="Image 2", image3_exp="Image 3", image4_exp="Image 4"):
#     f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 9))
#     f.tight_layout()
#     ax1.imshow(image1, cmap='gray')
#     ax1.set_title(image1_exp, fontsize=20)
#     ax2.imshow(image2, cmap='gray')
#     ax2.set_title(image2_exp, fontsize=20)
#     ax3.imshow(image3, cmap='gray')
#     ax3.set_title(image3_exp, fontsize=20)
#     ax4.imshow(image4, cmap='gray')
#     ax4.set_title(image4_exp, fontsize=20)
#     plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
#     plt.savefig("./output_images/feature_channelV.png")
#
# # show_images(car_ch1, car_hog_image, notcar_ch1, notcar_hog_image, "Car ch 1", "Car ch 1 HOG", "Not Car ch 1", "Not Car ch 1 HOG")
# # show_images(car_ch1, car_ch1_features, notcar_ch1, notcar_ch1_features, "Car ch Y", "Car ch Y features", "Not Car ch Y", "Not Car ch Y features")
# # show_images(car_ch2, car_ch2_features, notcar_ch2, notcar_ch2_features, "Car ch U", "Car ch U features", "Not Car ch U", "Not Car ch U features")
# show_images(car_ch3, car_ch3_features, notcar_ch3, notcar_ch3_features, "Car ch V", "Car ch V features", "Not Car ch V", "Not Car ch V features")


