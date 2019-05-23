# -*- coding: utf-8 -*-
"""
Created on Tue May 14 12:24:18 2019

@author: yujue
"""
import numpy as np
from sklearn.metrics import roc_curve
from slices_util import findSlicesCoordinate, findSlicesDistance, findClosestIndex 
from seg_util import createMask, extractBrainTissue
#from seg_util import thresholdDivision, largestConnectComponent
#from load_util import  loadDataFrame, loadContentList, loadPatientList
from load_util import loadImage, loadROI, loadImageArray, findRoiIndex


def prepareComparison(image_slice, mask_slice, thresh):
    '''get a single slice and prepare it for ROC analysis
    
    Arg:
        image_slice: single CTP slice
        mask_slice: single mask slice corresponding to CTP slice with ROI
        thresh
    
    Return:
        flat_dst: flatten the CTP slice masked by brain tissue
        flat_roi: flatten the mask slice masked by brain tissue
    '''
    
    # extract the brain tissue from the CTP slices
    brain = extractBrainTissue(image_slice)
    
    # get the mask of brain tissue
    labels = brain!=0
    
#    dst = thresholdDivision(brain, labels, thresh)
    dst = (brain <= thresh) * labels
    
    dst = dst.astype(int) # Optional

#    flat_dst = flattenArray(dst, labels)
#    flat_roi = flattenArray(mask_slice, labels)
    flat_dst = dst[labels != 0].flatten()
    flat_roi = mask_slice[labels != 0].flatten()
    assert(flat_dst.shape[0] == flat_roi.shape[0])
    
    return flat_dst, flat_roi

def flattenArray(dst, labels):
    '''flatten the array in terms of the position of brain tissue
    
    Arg:
        dst: single image slice array
        labels: brain tissue mask
        
    Return:
        flat: flattened image array in terms of the position of brain tissue
    
    '''
#    temp = np.copy(dst)
    assert(labels.shape == dst.shape)
#    flat = temp[labels != 0].flatten()
    flat = dst[labels != 0].flatten()
    assert(flat.shape[0] == np.sum(labels))
    return flat

def compareCTPandROI(flat_dsts, flat_rois):
    '''compare CTP and ROI using roi_curve using roc_curve
    
    Arg:
        flat_dsts
        flat_rois
    
    Return:
        fpr 
        tpr
    '''
    fpr,tpr,threshold = roc_curve(flat_rois, flat_dsts, pos_label=1)
    fpr = np.squeeze(fpr)
    tpr = np.squeeze(tpr)
    return fpr, tpr

def eachPatientSlice(image_array, roi_index, mask, raw_roi_index, thresh):
    '''processing a patient's whole profile with CTP and ROI
    
    Arg:
        image_array: CTP image_array
        roi_index: slice index of CTP image_array
        mask: ROI mask
        raw_roi_index: slice index of mask
        thresh
        
    return
    '''
    
    flat_dsts = np.array([])
    flat_rois = np.array([])
    assert(len(roi_index) == len(raw_roi_index))
    
    for i, r in enumerate(roi_index):
        
        # i is the index of raw_roi_index, containing the index of mask slice
        # r is the element of roi_index correpsonding to CTP image_array
        p = raw_roi_index[i]
        flat_dst, flat_roi = prepareComparison(image_array[r], mask[p], thresh)
        
        # concatenate the arrays of different slices from the same patient
        flat_dsts = np.concatenate((flat_dsts, flat_dst))
        flat_rois = np.concatenate((flat_rois, flat_roi))
    
    assert(flat_dsts.shape[0] == flat_rois.shape[0])
    return flat_dsts, flat_rois

def allPatient(content, patient_list, thresh,bSVD_df, ROI_df, CT_df):
    '''analyzing all profiles of patients
    
    Arg:
        content: different modes
        patient_list: list of patient IDs
        thresh
        bSVD_df: dataframe
        ROI_df: dataframe
        CT_df: dataframe
    
    Return:
        dsts: further concat the flat_dsts from single patients
        rois: same 
    '''
    dsts = np.array([])
    rois = np.array([])
    for i, patient in enumerate(patient_list):
        roi = loadROI(ROI_df.ROI_path[i]) # i
        raw_roi_index = findRoiIndex(roi)
        
        print('patient'+str(i))
       
        # some patient's ROI is empty
        if not raw_roi_index:
            continue
        elif i == 9 or i == 4 or i == 5: # index 9, 4, 5 patients are analmoly
            continue
        
        img_bSVD = loadImage(bSVD_df, content, patient)
   
        CTP_slices_coord = findSlicesCoordinate(img_bSVD['origin'][2], img_bSVD['spacing'][2], 
                                                img_bSVD['image_array'].shape[0])
       
        
        
        
        img_CT = loadImageArray(CT_df.loc[i, 'Patient_File'], # i
                                CT_df.loc[i, 'Series_ID'] )   # i
        
        roi_slices_coord = findSlicesCoordinate(img_CT['origin'][2], img_CT['spacing'][2],
                                              img_CT['image_array'].shape[0],
                                              is_roi=True, roi_slices_num=raw_roi_index)
       
        slices_dist = findSlicesDistance(CTP_slices_coord, roi_slices_coord)
        roi_index = findClosestIndex(slices_dist)
        mask = createMask(img_CT['image_array'], roi)
        
        flat_dsts, flat_rois = eachPatientSlice(img_bSVD['image_array'], 
                                                roi_index, mask, raw_roi_index, thresh)
        dsts = np.concatenate((dsts, flat_dsts))
        rois = np.concatenate((rois, flat_rois))
    return dsts,rois

