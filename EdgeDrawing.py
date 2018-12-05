# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 13:10:49 2018

@author: Shaojun Luo
"""
import numpy as np
import cv2

# define value of HORIZONTAL
HORIZONTAL = 1
VERTICAL = -1
LEFT = -1
RIGHT = 1
UP = -1
DOWN = 1
max_gap = 4
# default parameters
EDParam_default = {'ksize': 5,
                   'sigma': 1.0,
                   'gradientThreshold': 36, 
                   'anchorThreshold': 8,
                   'scanIntervals': 1}

class EdgeDrawing:
    # initiation 
    def __init__(self, EDParam = EDParam_default):
        #	EDLineDetector constructor function
        #set parameters for line segment detection
        self.ksize_ = EDParam['ksize']
        self.sigma_ = EDParam['sigma']
        self.gradientThreshold_ = EDParam['gradientThreshold']
        self.anchorThreshold_  = EDParam['anchorThreshold']
        self.scanIntervals_    = EDParam['scanIntervals']
        #dimension of image
        self.MAX_X = 0 
        self.MAX_Y = 0
        self.G_ = np.array([])
        self.D_ = np.array([])
        self.E_ = np.array([])
        
    # search algorithm for EdgeDrawing
    def GoUp_(self,x, y):
        segment = [] # array to record edge segment
        direct_next = None # search direction of left side similart to right, up and down  
        while x>0 and self.G_[x,y]>0 and not self.E_[x,y]:
            next_y = [max(0,y-1), y, min(self.MAX_Y-1,y+1)] # search in a valid area
            segment.append((x,y))# extend line segments
            if self.D_[x,y] == VERTICAL:
                self.E_[x, y] = True # mark as edge
                y_last = y # record parent pixel
                x, y = x-1, next_y[np.argmax(self.G_[x-1, next_y])]# walk to next pixel with max gradient
            else:
                direct_next = y - y_last # change direction to continue search
                break # stop and proceed to next search
        return segment,direct_next
        
    def GoDown_(self,x, y):
        segment = []
        direct_next = None
        while x < self.MAX_X-1 and self.G_[x,y]>0 and not self.E_[x, y]:
            next_y = [max(0,y-1), y, min(self.MAX_Y-1,y+1)]
            segment.append((x,y))
            if self.D_[x,y] == VERTICAL:
                self.E_[x, y] = True
                y_last = y
                x, y = x+1, next_y[np.argmax(self.G_[x+1, next_y])]
            else:
                direct_next = y - y_last
                break
        return segment,direct_next
            
    def GoRight_(self,x, y):
        segment = []
        direct_next = None
        while y < self.MAX_Y-1 and self.G_[x,y]>0 and not self.E_[x,y]:
            next_x =  [max(0,x-1), x, min(self.MAX_X-1,x+1)]
            segment.append((x,y))
            if self.D_[x,y] == HORIZONTAL:
                self.E_[x, y] = True
                x_last = x
                x, y = next_x[np.argmax(self.G_[next_x, y+1])], y+1
            else:
                direct_next = x - x_last
                break
        return segment,direct_next
    
    def GoLeft_(self,x, y):
        segment = []
        direct_next = None
        while y>0 and self.G_[x, y]>0 and not self.E_[x, y]:
            next_x = [max(0,x-1), x, min(self.MAX_X-1,x+1)]
            segment.append((x,y))
            if self.D_[x,y] == HORIZONTAL:
                self.E_[x, y] = True
                x_last = x
                x, y = next_x[np.argmax(self.G_[next_x, y-1])], y-1
            else:
                direct_next = x - x_last
                break
        return segment, direct_next
    
    # walk down until reach the end
    def SmartWalk_(self, x,y,direct_next):
        segment = [(x,y)]
        while direct_next is not None:
            x, y = segment[-1][0], segment[-1][1]
            # if the last point of chain is horizontal, explore horizontally
            if self.D_[x,y] == HORIZONTAL:
                # get segment sequence
                if direct_next == LEFT:
                    s, direct_next = self.GoLeft_(x,y)
                elif direct_next == RIGHT:
                    s, direct_next = self.GoRight_(x,y)
                else: break
#                    if self.G_[x,y+1]>self.G_[x,y-1]:
#                        s, direct_next = self.GoRight_(x,y)
#                    else:
#                        s, direct_next = self.GoLeft_(x,y)
            elif self.D_[x,y] == VERTICAL:  # explore vertically
                if direct_next == UP:
                    s, direct_next = self.GoUp_(x,y)
                elif direct_next == DOWN:
                    s, direct_next = self.GoDown_(x,y)
                else: break
#                    if self.G_[x-1,y]>self.G_[x+1,y]:
#                        s, direct_next = self.GoUp_(x,y)
#                    else:
#                        s, direct_next = self.GoDown_(x,y)
            else: # if the next pixel is invalid
                break
            if len(s) > 1:
                segment.extend(s[1:])
        return segment
    # find list of anchors
    def FindAnchors_(self,image):
        # detect the anchor
        anchor_list = []
        for i in range(1,image.shape[0]-1,self.scanIntervals_):
            for j in range(1,image.shape[1]-1,self.scanIntervals_):
                if self.D_[i,j] == HORIZONTAL:  # HORIZONTAL EDGl compare up & down
                    if self.G_[i,j] - self.G_[i-1,j] >= self.anchorThreshold_ and self.G_[i,j] - self.G_[i+1,j] >= self.anchorThreshold_:
                        anchor_list.append((i,j))
                elif self.D_[i,j] == VERTICAL: # VERTICAL EDGE. Compare with left & right.
                    if self.G_[i,j] - self.G_[i,j-1] >= self.anchorThreshold_ and self.G_[i,j] - self.G_[i,j+1] >= self.anchorThreshold_:                    
                        anchor_list.append((i,j))
        return anchor_list
    # merge edges
    def MergeEdges_(self,edges):
        # connect and merge the edges inplace
        merged = True
        while merged: # if last iteration perform merged
            p1 = 0 # pivot for first edge
            merged = False # assume not going to merge
            # iterate over edges to merge
            while p1 < len(edges):
                p2 = p1+1
                while p2 < len(edges):
                    # mark start and end point of 2 segments
                    start_1, end_1 = edges[p1][0], edges[p1][-1]
                    start_2, end_2 = edges[p2][0], edges[p2][-1]
                    # direction of two vectors
                    v_1 = (end_1[0]-start_1[0],end_1[1]-start_1[1])
                    v_2 = (end_2[0]-start_2[0],end_2[1]-start_2[1])
                    # if they aligned in the same direction, compare with head-end
                    if np.dot(v_1,v_2)>=0:
                        if abs(end_1[0] - start_2[0]) + abs(end_1[1] - start_2[1]) < max_gap:
                            # merge end-head
                            edges[p1] =  edges[p1] + edges.pop(p2)
                            merged = True
                        elif abs(start_1[0] - end_2[0])+abs(start_1[1] - end_2[1]) < max_gap:
                            # merge end-head
                            edges[p1] =  edges.pop(p2)+ edges[p1]
                            merged = True
                        else:
                            p2 += 1 # manually poceed to next segment
                    else:
                        if abs(start_1[0] - start_2[0]) + abs(start_1[1]-start_2[1]) < max_gap:
                            # merge head-head
                            edges[p1] = edges[p1][::-1] + edges.pop(p2)
                            merged = True
                        elif abs(end_1[0] - end_2[0]) + abs(end_1[1] - end_2[1]) < max_gap:
                            # merge end-end
                            edges[p1] = edges.pop(p2) + edges[p1][::-1]
                            merged = True
                        else:
                            p2 += 1 # manually poceed to next segment
                p1 += 1 # next segment
        return
    # edge drawing algorithm
    def EdgeDrawing(self, image, smoothed = False):
        # validation check for image
        if len(image.shape)>2:
            raise('Use only 1 channel or grayscale image')
            return None
        # set up dimension
        self.MAX_X = image.shape[0]
        self.MAX_Y = image.shape[1]
        # if not smoothed then smooth it
        if not smoothed: #input image hasn't been smoothed.
            img = cv2.GaussianBlur(image,(self.ksize_,self.ksize_),self.sigma_)
        else:
            img = image.copy()
        # compute dx,dy imagegradient
        dxImg_ = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=1)
        dyImg_ = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=1)
        
        # Compute gradient map and direction map
        #self.G_ = np.sqrt(dxImg_*dxImg_ + dyImg_*dyImg_)
        self.G_ = np.abs(dxImg_)+ np.abs(dyImg_)
        self.G_[self.G_ < self.gradientThreshold_] = 0  
        # If true, then it is horizontal edge
        self.D_ = -np.sign(np.abs(dxImg_) - np.abs(dyImg_))
        self.D_[self.G_ < self.gradientThreshold_] = 0
        
        #cv2.imwrite('Gradient.bmp',255*(self.G_>0).astype(int))
        # find anchor list
        anchor_list = self.FindAnchors_(image)                        
                
        edges = []
        # initiate edgemap
        self.E_ = np.zeros(self.G_.shape,dtype = bool)
        # first round edrawing, get fragment segments
        for anchor in anchor_list:
            if not self.E_[anchor]: # if not mark as edges
                # walk right or down
                segment_1 = self.SmartWalk_(anchor[0], anchor[1], 1)
                # reset anchor point
                self.E_[anchor] = False
                # walk left or up
                segment_2 = self.SmartWalk_(anchor[0], anchor[1], -1)
                # concat two segments
                if len(segment_1[::-1] + segment_2)>0:
                    edges.append(segment_1[::-1] + segment_2[1:])
        # merge the edges with same direction
        #self.MergeEdges_(edges)
        edge_map = 255*self.E_.astype(np.uint8)
        return edges, edge_map
    

         