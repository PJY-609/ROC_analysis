# -*- coding: utf-8 -*-
"""
Created on Fri May  3 20:13:29 2019

@author: yujue
"""

import numpy as np
import matplotlib.pyplot as plt
from constant import ITER
from load_util import  loadDataFrame, loadContentList, loadPatientList
from load_util import loadImage, loadROI, loadImageArray, findRoiIndex
from slices_util import findSlicesCoordinate, findSlicesDistance, findClosestIndex 
from seg_util import createMask, extractBrainTissue, labelConnectedArea
from seg_util import thresholdDivision, largestConnectComponent
from roc_util import allPatient, compareCTPandROI

INTERVALS ={
        'MTT-bSVD': [1, 20]
        }
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

def plotROCCurve(content, patient_list):
    fprs, tprs = analyzeParameters(INTERVALS[content], content, patient_list)
#    fprs.sort()
#    tprs.sort()
    roc = {'fprs':fprs,'tprs':tprs}
    np.save(content+'.npy', roc)
    plt.plot(fprs, tprs, color='darkorange', lw=1, label='ROC curve')

# In[]:
bSVD_df = loadDataFrame('bSVD')
patient_list = loadPatientList(bSVD_df)
plotROCCurve('MTT-bSVD', patient_list)
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