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
import glob
import time

# raw input 
#input_image = image

# ground truth folder
ground_truth_folder = 'test_data/IR_GT/'

# inflared Light
origin_image_folder = 'test_data/IR/'

# parameters for Edge Drawing
EDParam = {'ksize':3, # gaussian Smooth filter size if smoothed = False
           'sigma': 1, # gaussian smooth sigma ify smoothed = False
           'gradientThreshold': 25, # threshold on gradient image
           'anchorThreshold': 10, # threshold to determine the anchor
           'scanIntervals': 4} # scan interval, the smaller, the more detail
## Visible Light
#origin_image_folder = 'test_data/VL/'
#
## parameters for Edge Drawing (visible light)
#EDParam = {'ksize':3, # gaussian Smooth filter size if smoothed = False
#           'sigma': 1, # gaussian smooth sigma ify smoothed = False
#           'gradientThreshold': 25, # threshold on gradient image
#           'anchorThreshold': 10, # threshold to determine the anchor
#           'scanIntervals': 1} # scan interval, the smaller, the more detail

# image list
origin_image_list = sorted(glob.glob(origin_image_folder+'*.bmp'))
ground_truth_list = sorted(glob.glob(ground_truth_folder+'*.bmp'))
n = len(origin_image_list)

# initiate the EDLineDetector class
ED = EdgeDrawing.EdgeDrawing(EDParam)

""" 
Preprocess the images to get edges using EdgeDrawing
Also the test set is constructed for calculation of confusion matrix
"""
edges_list = []
edges_map_list = []
image_GT_list = []

start_time = time.time() # start timing
print('Preprocessing Image (EdgeDrawing)')

#Edge Drawing detection:
for i, image_file in enumerate(origin_image_list):
    # read image_file
    image = cv2.imread(image_file)
    # convert to gray-scale image
    input_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # edge Drawing
    edges,edges_map = ED.EdgeDrawing(input_image, smoothed = False)
    # edge list is a list of pixel chains it is used for EDLine and PLineD
    edges_list.append(edges)
    # edges map is a matrix with edges marked as 255, it is used for RHT
    edges_map_list.append(edges_map)
    #ground truth image
    image_ground_truth = cv2.imread(ground_truth_list[i])
    image_ground_truth = cv2.cvtColor(image_ground_truth,cv2.COLOR_BGR2GRAY) # convert to gray image
    image_GT_list.append(image_ground_truth)
    
    if (i+1)%100 == 0:
        print(str(i+1) + ' images processed')
        
elapsed_time = time.time() - start_time # end timming

# mean preprocess time for each image
mean_pre_time = elapsed_time/n

print('Preprocess Complete')
print('Time: {0:.1f}s. Average Time: {1:.3f}s'.format(elapsed_time, mean_pre_time))

#%%
""" 
Power Line detection Algorithm
"""

# parameters for RHT
distance_resolution = 1
angle_resolution = np.pi/180
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
tol_d = 3.0

print('---------Randomlized Hough------------')

TPR_list_HT = []
FPR_list_HT = []

start_time = time.time() # start timing
#
for i, edges_map in enumerate(edges_map_list):
    # RHT line detection
    lines = cv2.HoughLinesP(edges_map,distance_resolution,angle_resolution,votes,minLineLength,maxLineGap)
    lines = TT.TranslateLines(lines)
    # get prediction  label
    TP, FP, FN,_ = TT.ConfusionMatrix(lines,image_GT_list[i])
    if TP + FN > 4: # only count valid
        TPR_list_HT.append(TT.TPR(TP,FN))
    FPR_list_HT.append(TT.FPR(TP,FP))    
#
elapsed_time = time.time() - start_time # end time
# Note that the value of None will not count when calculate mean
print('Sensitivity: {0:.3f}'.format(np.array(TPR_list_HT).mean()))
print('Specificity: {0:.3f}'.format(1-np.array(FPR_list_HT).mean()))
print('Average time(including preprocessing): {0:.4f}s'.format(elapsed_time/n + mean_pre_time))

print('----------EDLine detection-------------')

TPR_list_ED = []
FPR_list_ED = []

start_time = time.time() # start timing

for i, edges in enumerate(edges_list):
    # EDline detection
    lines = LD.EDLine(edges, minLineLen,lineFitErrThreshold)
    # get prediction  label
    TP, FP, FN,_ = TT.ConfusionMatrix(lines,image_GT_list[i])
    if TP + FN > 4: #there is a coner dot wit 2x2 px in ground truth
        TPR_list_ED.append(TT.TPR(TP,FN))
    FPR_list_ED.append(TT.FPR(TP,FP))
    
elapsed_time = time.time() - start_time # end time

# when making the mean, it ignore the nan
print('Sensitivity: {0:.3f}'.format(np.array(TPR_list_ED).mean()))
print('Specificity: {0:.3f}'.format(1-np.array(FPR_list_ED).mean()))
print('Average time(including preprocessing): {0:.4f}s'.format(elapsed_time/n + mean_pre_time))

print('--------------PLineD---------------')

TPR_list_PD = []
FPR_list_PD = []

start_time = time.time() # start timing

for i, edges in enumerate(edges_list):
    # PLineD
    lines = LD.PLineD(edges,min_L, min_P,tol_a, tol_d)
    # get prediction  label
    TP, FP, FN, _ = TT.ConfusionMatrix(lines,image_GT_list[i])
    if TP > 4:
        TPR_list_PD.append(TT.TPR(TP,FN))
    FPR_list_PD.append(TT.FPR(TP,FP))

elapsed_time = time.time() - start_time # end time

print('Sensitivity: {0:.3f}'.format(np.array(TPR_list_PD).mean()))
print('Specificity: {0:.3f}'.format(1-np.array(FPR_list_PD).mean()))
print('Average time(including preprocessing): {0:.4f}s'.format(elapsed_time/n + mean_pre_time))