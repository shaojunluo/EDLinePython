# -*- coding: utf-8 -*-
"""
Created on Sun May 13 17:14:43 2018

@author: Shaojun Luo
"""
import numpy as np
from numpy.linalg import eig

# Internal Parameters
RATIO = 50
ANGLE_TURN = 67.5*np.pi/180
STEP = 3

""" 
Begin for EDLine Functions
"""
# distance from point to line
def Distance_(a,b,point):
    if b is None:
        return np.abs(point[0] - a)
    else:
        return np.abs(b*point[0]  - point[1] + a)/np.sqrt(b*b + 1)
    
# fast fit line equation
def FitLine_(pixel_chain):
    x,y = zip(*pixel_chain)
    x = np.float64(x)
    y = np.float64(y)
    if np.dot(x-x.mean(),x-x.mean()) == 0: #if it is horizontal line
        beta = None
        alpha = x.mean()
        mse = ((x - alpha)**2).mean()
    else: # else ordinary line, MSE take the orthogonal distance
        beta = np.dot(x-x.mean(),y-y.mean())/np.dot(x-x.mean(),x-x.mean())
        alpha = np.mean(y) - beta*np.mean(x)
        mse = np.array([Distance_(alpha,beta,point)**2 for point in pixel_chain]).mean()
    return beta, alpha, mse

# EDLine detection
def EDLine(edges, minLineLen, lineFitErrThreshold = 1):
    # filter edges
    edges = [edge for edge in edges if len(edge)>=minLineLen]
    lines = []
    # finding initial line segment
    while edges:
        edge = edges.pop(0)
        while len(edge) >= minLineLen:
            b,a,err = FitLine_(edge[:minLineLen])
            if err < lineFitErrThreshold: # initial segment fournd
                break
            else: # otherwise move the window
                edge = edge[1:]
                
        if err < lineFitErrThreshold: # if line segment found, extend the line
            # start from segment
            line_len = minLineLen
            while line_len < len(edge):#and err < self.lineFitErrThreshold_:
                if Distance_(a,b,edge[line_len]) <= lineFitErrThreshold:
                    line_len+= 1
                    #b,a,err = self.FitLine(edge[:line_len])
                else:
                    break
            lines.append(edge[:line_len])
            # append the rest of pixels for next line extraction
            if len(edge[line_len:])>= minLineLen:
                edges.append(edge[line_len:])
    return lines

""" 
Begin for PLineD Functions
"""
# cos of angle between two vectors
def cosAngle_(v_1,v_2):
    return np.dot(v_1,v_2)/np.sqrt(np.dot(v_1,v_1)*np.dot(v_2,v_2))

# check if is a line using covariant matrix
def CovarianceCheck_(segment):
    cov_mat = np.cov(list(zip(*segment))) # covariance matrix
    eig_val, _= eig(cov_mat) # eigen value
    #if the ratio is large then return true, 1e-10 is a protection on dividing 0
    return (max(eig_val) + 1e-10)/(min(eig_val) + 1e-10) > RATIO

# Algorithm 1&2 segment cut and line detection
def SegmentFilter_(edges):
    lines = []
    for edge in edges:
        while edge:
            if len(edge) > 2*STEP:
                i = STEP
                while i < len(edge)-STEP:
                    v_1 = np.subtract(edge[i],edge[i-STEP])
                    v_2 = np.subtract(edge[i+STEP],edge[i])
                    if cosAngle_(v_1,v_2) < np.cos(ANGLE_TURN):
                        break # find segment
                    else: # step foward
                        i = i+STEP
                if i > len(edge)-STEP: # if the search is already to end
                    segment = edge # attach whole segment
                    break
                else: # cut the segement
                    segment = edge[:i] # cut the current segment
                    edge = edge[i:] # proceed to next search
            else:
                segment = edge # attach whole segment
                break
            # check if the segment is a line 
            if CovarianceCheck_(segment):
                lines.append(segment)
    return lines

# Algorithm 3. Group the lines
def GroupLines_(lines, min_L, min_P, tol_a, tol_d):
        # connect and merge the line inplace
        line_groups = []
        v_groups = []
        p1 = 0
        while p1 < len(lines):
            if len(lines[p1])>min_L:# if the segment begin to consider
                L_line = lines.pop(p1) # pop the current segment
                max_l = len(L_line) # longest line segment
                group_ = [L_line] # initiate group
                p_group = len(L_line) # total pixel in segment
                v_1 = np.subtract(L_line[-1], L_line[0]) # v_1 is the longest segment direction
                v_m = v_1 # mean direction
                p2 = 0
                while p2 < len(lines):
                    v_s = np.subtract(lines[p2][-1], lines[p2][0]) # v_s is the direstion of new segment s
                    # if they aligned in the same direction, compare with head-end
                    if np.abs(cosAngle_(v_m,v_s)) > np.cos(tol_a):
                        # distance of two vectors
                        v_1_n = (-v_m[1],v_m[0]) # normal of mean-line
                        mid_2 = 0.5*np.add(lines[p2][0], lines[p2][-1]) # mid point of line s
                        v_2 = np.subtract(mid_2,L_line[-1]) # direction to any point on the line
                        ds = np.abs(np.dot(v_1_n,v_2))/np.sqrt(np.dot(v_1_n,v_1_n)) # distance of segment to line
                        # if they are close enough, merge
                        if ds < tol_d:
                            N_line = lines.pop(p2) # pop the current segment
                            group_.append(N_line) # add to group
                            p_group += len(N_line) # update group member
                            if max_l > len(N_line): # update longest segment
                                L_line = N_line
                                v_1 = np.subtract(L_line[-1], L_line[0])
                                v_m += v_1
                        else:
                            p2 += 1 # manually poceed to next segment
                    else:
                        p2 += 1 # manually poceed to next segment
                if p_group > min_P: # if the group is large enough
                    line_groups.append(group_) # put in the line group
                    v_groups.append(v_m) # main direction of this group
                p1 = 0 # if group then reset p1
            else:
                p1 += 1 # next segment
        return line_groups, v_groups

# Algorithm 4: detect parallel groups
def ParallelGroups_(line_groups,v_groups, tol_a):
    parallel_groups = []
    while line_groups:
        l_1 = line_groups.pop(0) # pop the first element
        v_1 = v_groups.pop(0) 
        parallel_segments = [l_1] # initiate parallel segments
        p = 0 
        while p < len(line_groups):
            v_s = v_groups[p]
            if np.abs(cosAngle_(v_1,v_s)) > np.cos(tol_a):
                #add the segment to group and remove from pool
                parallel_segments.append(line_groups.pop(p)) 
                v_groups.pop(p)
            else:
                p += 1 # manually proceed
        if len(parallel_segments) > 1: # there are parallel groups
            for segments in parallel_segments:
                for s in  segments:
                    parallel_groups.append(s)
    # check the group
    return parallel_groups

# main body for PlineD
def PLineD(edges, min_L = 10, min_P = 1000, tol_a = 5*np.pi/180, tol_d = 60):
    # cut and filter line segments
    lines = SegmentFilter_(edges)
    #
    # group lines
    line_groups, v_groups = GroupLines_(lines, min_L, min_P, tol_a, tol_d)
    #return line_groups,v_groups
    # find parallel lines
    parallel_groups = ParallelGroups_(line_groups,v_groups, tol_a)
    # return
    return parallel_groups
        