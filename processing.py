# -*- coding: utf-8 -*-
"""
Created on Fri May  3 20:13:29 2019

@author: yujue
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from constant import ITER
from load_util import  loadDataFrame, loadContentList, loadPatientList
from load_util import loadImage, loadROI, loadImageArray
   
def createMask(image_array, roi):
    mask = np.zeros([image_array.shape[0], image_array.shape[1], image_array.shape[2]])
    for k in roi.keys():
        point_sets = [np.array(x) for x in roi[k]]
        for p in range(len(point_sets)):
            point_set = np.int32([point_sets[p]])
            cv2.fillConvexPoly(mask[k], point_set, 1)
    return mask        

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

from skimage import data,filters,segmentation,measure,morphology,color
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
    dst =np.multiply((brain >= thresh), labels)
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

from sklearn.metrics import roc_curve 
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
    

def analyzeParameters(interval, content, patient_list):
    increment = int((interval[1] - interval[0]) / ITER)
    fprs = np.array([])
    tprs = np.array([])
    
    bSVD_df = loadDataFrame('bSVD')
    ROI_df = loadDataFrame('ROI')
    CT_df = loadDataFrame('CT')
    for i in range(ITER):
        print('iteration'+str(i))
        thresh = interval[0] + i * increment 
#        flat_dsts, flat_rois = eachPatientSlice(image_array, roi_index_list, mask, thresh)
        dsts, rois = allPatient(content, patient_list, thresh, bSVD_df, ROI_df, CT_df)
        fpr, tpr = compareCTPandROI(dsts, rois)
        fpr = np.array([fpr[1]])
        tpr = np.array([tpr[1]])
        fprs = np.concatenate((fprs, fpr))
        tprs = np.concatenate((tprs, tpr))

    return fprs, tprs

def findRoiIndex(roi):
    roi_index=[]
    for i in roi.keys():
        if roi[i]:
            roi_index.append(i)
    return roi_index


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

def plotROCCurve(content, patient_list):
    fprs, tprs = analyzeParameters(intervals[content], content, patient_list)
#    fprs.sort()
#    tprs.sort()
    roc = {'fprs':fprs,'tprs':tprs}
    np.save(content+'.npy', roc)
    plt.plot(fprs, tprs, color='darkorange', lw=1, label='ROC curve')

intervals ={
        'MTT-bSVD': [1, 20]
        }

# In[]:
bSVD_df = loadDataFrame('bSVD')
patient_list = loadPatientList(bSVD_df)
#plotROCCurve('MTT-bSVD', patient_list)
# In[]:
CT_df = loadDataFrame('CT')
ROI_df = loadDataFrame('ROI')
#for i, patient in enumerate(patient_list):
img_CT = loadImageArray(CT_df.loc[1, 'Patient_File'], 
                                CT_df.loc[1, 'Series_ID'] )
img_bSVD = loadImage(bSVD_df, 'MTT-bSVD', 'NCT143993')
CTP_slices_coord = findSlicesCoordinate(img_bSVD['origin'][2], img_bSVD['spacing'][2], 
                                            img_bSVD['image_array'].shape[0])
roi = loadROI(ROI_df.ROI_path[1])
raw_roi_index = findRoiIndex(roi)
roi_slices_coord = findSlicesCoordinate(img_CT['origin'][2], img_CT['spacing'][2],
                                              img_CT['image_array'].shape[0],
                                              is_roi=True, roi_slices_num=raw_roi_index)
mask = createMask(img_CT['image_array'], roi)
slices_dist = findSlicesDistance(CTP_slices_coord, roi_slices_coord)
roi_index = findClosestIndex(slices_dist)
# In[]:
for i in range(10):
    thresh = 1 + i * 1.9
    for i,p in enumerate(roi_index):
        
        brain = extractBrainTissue(img_bSVD['image_array'][p])
        labels = brain!=0
        dst =thresholdDivision(brain, labels, thresh)
        
        plt.figure(), plt.imshow(dst, 'gray',)
        
        

# In[]:
for i, patient in enumerate(patient_list):
    
    print(patient,img_bSVD['origin'], img_bSVD['image_array'].shape[0])
    
# In[]:
roi = loadROI(ROI_df.ROI_path[8])
print(not roi)