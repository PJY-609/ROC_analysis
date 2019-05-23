# -*- coding: utf-8 -*-
"""
Created on Tue May 14 12:19:16 2019

@author: yujue
"""
import cv2
from skimage import filters,measure,morphology
import numpy as np


def createMask(image_array, roi):
    '''create mask of ROI 
    
    Arg:
        images_array: CT image_array
        roi: dict such as, {16:[[1,1], [2, 2], [3, 3]], 17:[[4, 4]], [[5, 5]]}
    
    Return:
        mask: corresponding to CT image_array
    '''
    
    mask = np.zeros([image_array.shape[0], image_array.shape[1], image_array.shape[2]])
    
    for k in roi:
        
        '''there might be multiple ROI point sets
        e.g. roi[k]: [[[2, 2]], [[3, 3]]]
        point_sets: [np.array([[2, 2]]), np.array([[3, 3]])]
        point_set； np.array([[2, 2]])
        ''' 
        point_sets = [np.array(x) for x in roi[k]]
        
        for p in range(len(point_sets)):
            point_set = np.int32([point_sets[p]])
            
            # the keys of roi needs to be subtractd by 1
            cv2.fillConvexPoly(mask[k - 1], point_set, 1)
    
    return mask      


def extractBrainTissue(image_slice):
    '''extract the brain tissue from CTP
    
    Arg:
        image_slice: single slice
    
    Return:
        brain_slice: only with brain tissue
    '''
    img_mask = (image_slice != 0)

    lcc = largestConnectComponent(img_mask)
    brain_slice = np.multiply(image_slice, lcc)
    return brain_slice


def labelConnectedArea(image_slice):
    '''get diffetent connected area labeled
    
    Arg:
        image_slice: single slice
        
    Return:
        labels: different connected area with respective labels
        num: num of labels
    '''
    
    # find the threshhold using Otsu's method
    thresh = filters.threshold_otsu(image_slice)
    
    # closing based on Otsu's threshhold
    bw = morphology.closing(image_slice > thresh, morphology.disk(1))
    
    # get rid of bones still connected to brain tissue
    bw = morphology.opening(bw,morphology.disk(9)) 
    
    # get labeled
    labels, num = measure.label(bw,connectivity=1, background=0, return_num=True)
    
    return labels, num

def thresholdDivision(image_slice, labels, thresh):
    '''threshhold division with a mask
    
    Arg:
        image_slice: slice waits to be divided
        thresh: threshold
        labels: brain tissue mask
        
    Return:
        dst: division result
    
    '''
    dst = np.multiply((image_slice >= thresh), labels)
    return dst

def largestConnectComponent(bw_img, ):
    '''
    compute largest Connect component of an labeled image

    Parameters:
    ---

    bw_img:
        binary image

    Example:
    ---
        >>> lcc = largestConnectComponent(bw_img)

    '''

#    labeled_img, num = measure.label(bw_img, neighbors=4, background=0, return_num=True)    
    # plt.figure(), plt.imshow(labeled_img, 'gray')
    labeled_img, num = labelConnectedArea(bw_img)
    max_label = 0
    max_num = 0
    for i in range(1, num + 1): # 这里从1开始，防止将背景设置为最大连通域
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    lcc = (labeled_img == max_label)

    return lcc
