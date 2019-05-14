# -*- coding: utf-8 -*-
"""
Created on Tue May 14 12:19:16 2019

@author: yujue
"""
import cv2
from skimage import filters,measure,morphology
import numpy as np

def createMask(image_array, roi):
    mask = np.zeros([image_array.shape[0], image_array.shape[1], image_array.shape[2]])
    for k in roi.keys():
        point_sets = [np.array(x) for x in roi[k]]
        for p in range(len(point_sets)):
            point_set = np.int32([point_sets[p]])
            cv2.fillConvexPoly(mask[k], point_set, 1)
    return mask      


def extractBrainTissue(image_slice):
    img_mask = image_slice != 0
    labels = labelConnectedArea(img_mask)
    lcc = largestConnectComponent(labels)
    brain_slice = np.multiply(image_slice, lcc)
    return brain_slice

def labelConnectedArea(image_slice):
    thresh = filters.threshold_otsu(image_slice)
    bw =morphology.closing(image_slice > thresh, morphology.square(3))
    labels =measure.label(bw)
    return labels

def thresholdDivision(image_slice, labels, thresh):
    dst =np.multiply((image_slice >= thresh), labels)
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

    labeled_img, num = measure.label(bw_img, neighbors=4, background=0, return_num=True)    
    # plt.figure(), plt.imshow(labeled_img, 'gray')

    max_label = 0
    max_num = 0
    for i in range(1, num): # 这里从1开始，防止将背景设置为最大连通域
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    lcc = (labeled_img == max_label)

    return lcc
