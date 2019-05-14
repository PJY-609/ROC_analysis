# -*- coding: utf-8 -*-
"""
Created on Tue May 14 12:24:18 2019

@author: yujue
"""
import numpy as np
from sklearn.metrics import roc_curve
from slices_util import findSlicesCoordinate, findSlicesDistance, findClosestIndex 
from seg_util import createMask, extractBrainTissue, labelConnectedArea
from seg_util import thresholdDivision, largestConnectComponent
from load_util import  loadDataFrame, loadContentList, loadPatientList
from load_util import loadImage, loadROI, loadImageArray, findRoiIndex
from constant import ITER

def prepareComparison(image_slice,mask_slice, thresh):
    brain = extractBrainTissue(image_slice)
#    labels = labelConnectedArea(brain)
    labels = brain!=0
    dst = thresholdDivision(brain, labels, thresh)
    flat_dst = flattenArray(dst, labels)
    flat_roi = flattenArray(mask_slice, labels)
    assert(flat_dst.shape[0] == flat_roi.shape[0])
    return flat_dst, flat_roi

def flattenArray(dst, labels):
    temp = np.copy(dst)
    assert(labels.shape == dst.shape)
    flat = temp[labels != 0].flatten()
    assert(flat.shape[0] == np.sum(labels))
    #flat = flat.reshape(1, np.sum(labels))
    return flat

def compareCTPandROI(flat_dsts, flat_rois):
    fpr,tpr,threshold = roc_curve(flat_rois, flat_dsts, pos_label=1)
    fpr = np.squeeze(fpr)
    tpr = np.squeeze(tpr)
    return fpr, tpr

def eachPatientSlice(image_array, roi_index, mask, raw_roi_index, thresh):
    flat_dsts = np.array([])
    flat_rois = np.array([])
    assert(len(roi_index) == len(raw_roi_index))
    for i, r in enumerate(roi_index):
        flat_dst, flat_roi = prepareComparison(image_array[r], mask[raw_roi_index[i]], thresh)
        flat_dsts = np.concatenate((flat_dsts, flat_dst))
        flat_rois = np.concatenate((flat_rois, flat_roi))
    assert(flat_dsts.shape[0] == flat_rois.shape[0])
    return flat_dsts, flat_rois

def allPatient(content, patient_list, thresh,bSVD_df, ROI_df, CT_df):
    dsts = np.array([])
    rois = np.array([])
    for i, patient in enumerate(patient_list):
        img_bSVD = loadImage(bSVD_df, content, patient)
    #    
        CTP_slices_coord = findSlicesCoordinate(img_bSVD['origin'][2], img_bSVD['spacing'][2], 
                                                img_bSVD['image_array'].shape[0])
        roi = loadROI(ROI_df.ROI_path[i])
        raw_roi_index = findRoiIndex(roi)
        
        if not raw_roi_index:
            continue
        
        img_CT = loadImageArray(CT_df.loc[i, 'Patient_File'], 
                                CT_df.loc[i, 'Series_ID'] )
        
        roi_slices_coord = findSlicesCoordinate(img_CT['origin'][2], img_CT['spacing'][2],
                                              img_CT['image_array'].shape[0],
                                              is_roi=True, roi_slices_num=raw_roi_index)
       
        slices_dist = findSlicesDistance(CTP_slices_coord, roi_slices_coord)
        roi_index = findClosestIndex(slices_dist)
        mask = createMask(img_CT['image_array'], roi)
#        plotROCCurve('MTT', img_bSVD['image_array'], roi_list, mask)
        flat_dsts, flat_rois = eachPatientSlice(img_bSVD['image_array'], 
                                                roi_index, mask, raw_roi_index, thresh)
        dsts = np.concatenate((dsts, flat_dsts))
        rois = np.concatenate((rois, flat_rois))
    return dsts,rois


