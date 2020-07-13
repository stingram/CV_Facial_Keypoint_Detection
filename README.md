[//]: # (Image References)

[image1]: ./images/key_pts_example.png "Facial Keypoint Detection"



# CV_Facial_Keypoint_Detection
Implementation of a Facial Keypoint Detector with a Neural Network


## Summary

In this project, I combine computer vision techniques and deep learning architectures to build a facial keypoint detection system that takes in any image with faces, and predicts the location of 68 distinguishing keypoints on each face.

Facial keypoints include points around the eyes, nose, and mouth on a face and are used in many applications. These applications include: facial tracking, facial pose recognition, facial filters, and emotion recognition. My code, broken up into various notebooks show that the code is  able to look at any image, detect faces, and predict the locations of facial keypoints on each face. Some examples of these keypoints are pictured below.

![Facial Keypoint Detection][image1]

## Network Architecture

### Description
I started with a series of Convolution, activation, maxpooling layers followed by fully connected layers. I did added more convolutional layers as well as dropout to help avoid overfitting. For the loss function I chose Mean Square Error because the network is predicting a continuous set of values, as opposed to distinct categories. For the optimizer I chose Adam because it uses an adaptive learning rate. The detailed model description is below.

### Model
```
Net(
  (conv): Sequential(
    (0): Conv2d(1, 32, kernel_size=(5, 5), stride=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1))
    (4): ReLU()
    (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))
    (7): ReLU()
    (8): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (9): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1))
    (10): ReLU()
  )
  (fc): Sequential(
    (0): Linear(in_features=135424, out_features=512, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.25)
    (3): Linear(in_features=512, out_features=256, bias=True)
    (4): ReLU()
    (5): Dropout(p=0.2)
    (6): Linear(in_features=256, out_features=136, bias=True)
  )
)
```

## How to use this repository

Simply go through the notebooks in the order specified by the first number given at the beginning of each notebook name. For reference, they are listed here:

```
1. Load and Visualize Data.ipynb
2. Define the Network Architecture.ipynb
3. Facial Keypoints Detection, Complete Pipeline.ipynb
4. Fun with Keypoints.ipynb
```