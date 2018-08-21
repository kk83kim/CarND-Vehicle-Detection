## Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car.png
[image2]: ./output_images/not_car.png
[image3]: ./output_images/feature_hog.png
[image4]: ./output_images/feature_channelY.png
[image5]: ./output_images/feature_channelU.png
[image6]: ./output_images/feature_channelV.png
[image7]: ./output_images/windows.png
[image8]: ./output_images/test4.png
[image9]: ./output_images/heatmap2.png
[image10]: ./output_images/label2.png
[video1]: ./project_video_out.mp4


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt text][image2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.  The code for hog feature, binned color feature, and color histogram feature are in feature_function.py.

Here is an example using the `YUV` color space and HOG parameters of `orientations=15`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]

#### 2. Explain how you settled on your final choice of HOG parameters.

This was based on trial and error.  I tested with various combination of parameters and finally chose the above mentioned values because they seemed to give good results, but not too computationally heavy.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the sklearn.svm.LinearSVC().  I trained with all of the given dataset, including the Kitti dataset.  Before training, I normalized the dataset and shuffled so that the images were not in order.  The code for this section is in train_model.py

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I only searched the bottom half of the image since vehicles are most likely to be on the road.  Since objects further away are smaller than the near, I used smaller scale on the middle of the image and bigger scale towards the bottom of the image.  I chose the amount of the overlap, again with trial and error.  I chose the amount of overlap so that it is large enough to differentiate multiple vehicles that are close, but small enough to appropriately draw boundary box around the detected vehicles.  The code for this step is in apply_sliding_window function in main.py

![alt text][image7]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are an example images:

![alt text][image8]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections, I created a heatmap, added to the history (always kept accumulated heatmap from the last 5 frames), and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

### Classifier result and Heatmap
![alt text][image9]

### Label and Boundary Boxes on Detected Vehicle
![alt text][image10]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Although my current implementation works quite well on the given test video, there are still room for improvement such as...

Fine tuning: There are many knobs that can be tuned to improve performance such as selection of color spaces, hog parameters, SVM parameters for model fitting, sliding window scale / overlapping, etc.

Sanity Check: Current implementation stores heatmap from last 5 frames and threshold to finally find label and bounding box. One way to improve this is to do sanity check on the incoming heatmap of the current frame and discard if they deviate too much from the past. Even better wasy would be to weigh the heatmaps differently based on some factors, such as probability of classification.

Tracking with Prediction:  By keeping track of the detected vehicle, we could predict where the detected vehicle would be in the next frame and fuse the prediction with the measurement.

