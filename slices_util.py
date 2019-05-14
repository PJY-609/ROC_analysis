# -*- coding: utf-8 -*-
"""
Created on Tue May 14 12:16:02 2019

@author: yujue
"""

import numpy as np

def findSlicesCoordinate(origin_z, spacing_z, slice_num, is_roi=False, roi_slices_num=None):
    slices_coord = np.array([origin_z + spacing_z * i for i in range(slice_num)]).reshape(1,slice_num)
    if is_roi and roi_slices_num:
        slices_coord = np.array([slices_coord[0][x] for x in roi_slices_num]).reshape(1,len(roi_slices_num)) 
    return slices_coord    

def findSlicesDistance(CTP_slices_coord, roi_slices_coord):
    num_roi = roi_slices_coord.shape[1]
    num_slices = CTP_slices_coord.shape[1]
    slices_coord = CTP_slices_coord * np.ones((num_roi, num_slices))
    slices_dist = np.abs(slices_coord - roi_slices_coord.T)
    return slices_dist

def findClosestIndex(slices_dist):
    assert(isinstance(slices_dist, np.ndarray))
    assert(slices_dist.size != 0)
    index=[]
    for i in range(slices_dist.shape[0]):
        dist_list = list(slices_dist[i])
        index.append(dist_list.index(min(dist_list)))
    return index