# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 23:15:02 2019

@author: Utkarsh
"""

import cv2
import os
os.chdir('F:\\Python\\Open_CV\\Programs_Open_CV')

cap=cv2.VideoCapture('people-walking.mp4')
fgbg=cv2.createBackgroundSubtractorMOG2()

while True:
    ret,frame=cap.read()
    fgmask=fgbg.apply(frame)
    
    cv2.imshow('original',frame)
    cv2.imshow('fgbg',fgmask)
    
    k=cv2.waitKey(5) & 0xff
    if k==27:
        break

cap.release()
cv2.destroyAllWindows()
    