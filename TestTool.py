# -*- coding: utf-8 -*-

import numpy as np
import cv2

THRES = 20
K = 5
SIGMA = 3

# true positive Rate (sensitivity)
def TPR(TP,FN):
    if TP + FN > 0:
        return float(TP)/(TP+FN)
    else:
        return None
  # false positive rate (1- specificity)  
def FPR(TP, FP):
    if FP+TP > 0:
        return float(FP)/(TP+FP)
    else:
        return 0.0
# traslate RHT lines to pixel chains
def TranslateLines(RHT_lines):
    lines = []
    if RHT_lines is not None: # if valid
        for line in RHT_lines:
            line = line[0]
            # longest direction
            n = max(abs(line[2]-line[0]+1), abs(line[3]-line[1]+1))
            # expadn line
            x = np.round(np.linspace(line[0],line[2],n)).astype(int).tolist()
            y = np.round(np.linspace(line[1],line[3],n)).astype(int).tolist()
            lines.append(list(zip(*[y,x])))
    return lines

# Confusion matrix
def ConfusionMatrix(pred_lines, ground_truth):
    # concatenate all pixel points
    if pred_lines:
        pixels = np.concatenate(pred_lines)
    else:
        pixels = []
    # create edge map
    edge_map = np.zeros(ground_truth.shape)
    edge_map[list(zip(*pixels))] = 255
    # smooth the map to get confidence
    edge_map = cv2.GaussianBlur(edge_map,(K,K),SIGMA)
    # calculate overlap
    TP = np.sum(np.logical_and(ground_truth > THRES, edge_map > THRES))
    # False Positive
    FP = np.sum(np.logical_and(ground_truth < THRES, edge_map > THRES))
    # False Negative
    FN = np.sum(np.logical_and(ground_truth > THRES, edge_map < THRES))
    
    return TP, FP, FN, edge_map