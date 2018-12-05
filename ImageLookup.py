# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 12:09:25 2018

@author: Shaojun Luo
"""

import EdgeDrawing
import LineDetector as LD
import TestTool as TT
import cv2
import numpy as np

# raw input 
#input_image = image

# parameters
EDParam = {'ksize':3, # gaussian Smooth filter size if smoothed = False
           'sigma': 1, # gaussian smooth sigma ify smoothed = False
           'gradientThreshold': 25, # threshold on gradient image
           'anchorThreshold': 10, # threshold to determine the anchor
           'scanIntervals': 4} # scan interval, the smaller, the more detail

# parameters for RHT
distance_resolution = 1
angle_resolution = 0.5*np.pi/180
minLineLength = 12
maxLineGap = 10
votes = 20

# Parameters for EDLine
minLineLen = 40
lineFitErrThreshold = 1.0

# parameters for PLineD
min_L = 5
min_P = 20
tol_a = 5*np.pi/180
tol_d = 1.0

image_number = '00176'
# initiate the EDDrawing class
ED = EdgeDrawing.EdgeDrawing(EDParam)
# import image
image = cv2.imread('test_data/VL/TV_VL_ORG_'+image_number+'.bmp')
# ground Truth
image_ground_truth = cv2.imread('test_data/IR_GT/TV_IR_GT_'+image_number+'.bmp')
ground_truth = cv2.cvtColor(image_ground_truth,cv2.COLOR_BGR2GRAY)
# convert to gray-scale image
input_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#sharpen when needed
#ret,input_image = cv2.threshold(input_image,127,255,cv2.THRESH_TRUNC)

# edge Drawing
edges,edges_map = ED.EdgeDrawing(input_image, smoothed = False)

# RHT
lines = cv2.HoughLinesP(edges_map,distance_resolution,angle_resolution,votes,minLineLength,maxLineGap)
lines = TT.TranslateLines(lines)
# visualize
_,_,_,lines_map = TT.ConfusionMatrix(lines, ground_truth)
cv2.imwrite('lines_RHT.bmp',lines_map)# save  

# EDLine
lines = LD.EDLine(edges, minLineLen,lineFitErrThreshold)
_,_,_,lines_map = TT.ConfusionMatrix(lines, ground_truth)
cv2.imwrite('lines_EDLine.bmp',lines_map)# save  

# PLineD
lines = LD.PLineD(edges,min_L, min_P,tol_a, tol_d = tol_d)
lines_map = np.zeros(image.shape) # blank
_,_,_,lines_map = TT.ConfusionMatrix(lines, ground_truth)
cv2.imwrite('lines_PlineD.bmp',lines_map)

# save the original image and ground truth
cv2.imwrite('lines_Origin.bmp',image)
cv2.imwrite('lines_GroundTruth.bmp',image_ground_truth)

print('Complete')