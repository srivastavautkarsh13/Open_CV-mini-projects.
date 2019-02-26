# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 04:07:40 2019

@author: Utkarsh
"""
# # Dense Optical Flow in OpenCV
# 
# calcOpticalFlowFarneback(prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags) -> flow
# 
# This function computes a dense optical flow using the Gunnar Farneback's algorithm.
# 
# Here are the parameters for the function and what they represent:
#    

# * prev first 8-bit single-channel input image.
# * next second input image of the same size and the same type as prev.
# * flow computed flow image that has the same size as prev and type CV_32FC2.
# * pyr_scale parameter, specifying the image scale (\<1) to build pyramids for each image
#     * pyr_scale=0.5 means a classical pyramid, where each next layer is twice smaller than the previous one.
#     
# * levels number of pyramid layers including the initial image; levels=1 means that no extra layers are created and only the original images are used.
# * winsize averaging window size
#     * larger values increase the algorithm robustness to image
# * noise and give more chances for fast motion detection, but yield more blurred motion field.
# * iterations number of iterations the algorithm does at each pyramid level.
# * poly_n size of the pixel neighborhood used to find polynomial expansion in each pixel
#     * larger values mean that the image will be approximated with smoother surfaces, yielding more robust algorithm and more blurred motion field, typically poly_n =5 or 7.
# * poly_sigma standard deviation of the Gaussian that is used to smooth derivatives used as a basis for the polynomial expansion; for poly_n=5, you can set poly_sigma=1.1, for poly_n=7, a good value would be poly_sigma=1.5.

# In[6]:

import cv2 
import numpy as np

# Capture the frame
cap = cv2.VideoCapture(0)
ret, frame1 = cap.read()

# Get gray scale image of first frame and make a mask in HSV color
prvsImg = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

hsv_mask = np.zeros_like(frame1)
hsv_mask[:,:,1] = 255

while True:
    ret, frame2 = cap.read()
    nextImg = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    
    # Check out the markdown text above for a break down of these paramters, most of these are just suggested defaults
    flow = cv2.calcOpticalFlowFarneback(prvsImg,nextImg, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    
    # Color the channels based on the angle of travel
    # Pay close attention to your video, the path of the direction of flow will determine color!
    mag, ang = cv2.cartToPolar(flow[:,:,0], flow[:,:,1],angleInDegrees=True)
    hsv_mask[:,:,0] = ang/2
    hsv_mask[:,:,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    
    # Convert back to BGR to show with imshow from cv
    bgr = cv2.cvtColor(hsv_mask,cv2.COLOR_HSV2BGR)
    cv2.imshow('frame2',bgr)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    
    # Set the Previous image as the next iamge for the loop
    prvsImg = nextImg

    
cap.release()
cv2.destroyAllWindows()
