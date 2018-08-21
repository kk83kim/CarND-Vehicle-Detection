import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from feature_functions import *
import pickle


# Read in cars
cars = []
images = glob.glob('./vehicles/GTI_Far/image*.png')
for image in images:
    cars.append(image)
images = glob.glob('./vehicles/GTI_Left/image*.png')
for image in images:
    cars.append(image)
images = glob.glob('./vehicles/GTI_MiddleClose/image*.png')
for image in images:
    cars.append(image)
images = glob.glob('./vehicles/GTI_Right/image*.png')
for image in images:
    cars.append(image)
images = glob.glob('./vehicles/KITTI_extracted/*.png')
for image in images:
    cars.append(image)

# Read in not cars
notcars = []
images = glob.glob('./non-vehicles/GTI/image*.png')
for image in images:
    notcars.append(image)
images = glob.glob('./non-vehicles/Extras/extra*.png')
for image in images:
    notcars.append(image)

color_space = 'YUV'         # BGR, HSV, LUV, HLS, YUV, YCrCb
orient = 15                 # HOG orientations
pix_per_cell = 8            # HOG pixels per cell
cell_per_block = 2          # HOG cells per block
hog_channel = 'ALL'         # 0, 1, 2, or "ALL"
spatial_size = (32, 32)     # Spatial binning dimensions
hist_bins = 32              # Number of histogram bins
spatial_feat = True         # Spatial features on or off
hist_feat = True            # Histogram features on or off
hog_feat = True             # HOG features on or off

t = time.time()
car_features = extract_features(cars, color_space=color_space,
                                spatial_size=spatial_size, hist_bins=hist_bins,
                                orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block,
                                hog_channel=hog_channel, spatial_feat=spatial_feat,
                                hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space,
                                   spatial_size=spatial_size, hist_bins=hist_bins,
                                   orient=orient, pix_per_cell=pix_per_cell,
                                   cell_per_block=cell_per_block,
                                   hog_channel=hog_channel, spatial_feat=spatial_feat,
                                   hist_feat=hist_feat, hog_feat=hog_feat)
print(round(time.time() - t, 2), 'Feature Extraction')


t = time.time()
# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

# Fit a per-column scaler
X_scaler = StandardScaler().fit(X_train)
# Apply the scaler to X
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)
print(round(time.time() - t, 2), 'Process Data')
print('Using:', orient, 'orientations', pix_per_cell, 'pixels per cell and', cell_per_block, 'cells per block')
print('Feature vector length:', len(X_train[0]))

# Use a linear SVC
clf = LinearSVC()
t = time.time()
# Train model
clf.fit(X_train, y_train)
print(round(time.time() - t, 2), 'SVC Train')
print('Train Accuracy = ', round(clf.score(X_train, y_train), 4))
print('Test Accuracy = ', round(clf.score(X_test, y_test), 4))

# Save model
model = {"clf": clf, "scaler": X_scaler, "orient": orient,  "pix_per_cell": pix_per_cell,  "cell_per_block": cell_per_block,
         "spatial_size": spatial_size,  "hist_bins": hist_bins}
pickle.dump(model, open("model.p", "wb"))







