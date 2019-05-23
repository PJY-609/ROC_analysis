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
        'MTT-bSVD': [0, 550],
        'TTP': [0, 800],
        'CBV-bSVD': [0, 20],
        'CBV2-bSVD': [0,500],
#        'CBV-AUC': [0, 10],
        'CBF-bSVD':[0, 200],
        'CBF2-bSVD': [0, 100]
        }
def analyzeParameters(interval, content, patient_list):
    increment = (interval[1] - interval[0]) / ITER
    fprs = np.array([])
    tprs = np.array([])
    
    bSVD_df = loadDataFrame('bSVD')
    ROI_df = loadDataFrame('ROI')
    CT_df = loadDataFrame('CT')
    for i in range(ITER):
        thresh = interval[0] + i * increment 
#        flat_dsts, flat_rois = eachPatientSlice(image_array, roi_index_list, mask, thresh)
        dsts, rois = allPatient(content, patient_list, thresh, bSVD_df, ROI_df, CT_df)
        fpr, tpr = compareCTPandROI(dsts, rois)
        if len(fpr) == 2 and len(tpr) ==2:
            fpr = np.array([0])
            tpr = np.array([0])
        else:
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
    np.save(content+'.npz', roc)
    plt.plot(fprs, tprs, color='darkorange', lw=1, label='ROC curve')

# In[]:
bSVD_df = loadDataFrame('bSVD')
patient_list = loadPatientList(bSVD_df)
print(patient_list)
# In[]:
plotROCCurve('CBF2-bSVD',patient_list)
# In[]:
from sklearn import metrics
auc = {}
for mode in ['CBF-bSVD', 'CBF2-bSVD']:
    roc = np.load('{}.npz.npy'.format(mode))
    roc = roc.all()
    fprs = roc['fprs']
    tprs = roc['tprs']
    print(fprs)
    print(tprs)
    auc.update({mode: metrics.auc(fprs, tprs)})
    plt.plot(fprs, tprs, label=mode + '(AUC = ' + str(round(auc[mode],4)) + ')')
    
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.legend(loc='lower right')
plt.show()
max(auc, key=auc.get)
print(auc)
#pair = [(fprs[i], tprs[i]) for i in range(len(fprs))]
#pair.sort()
#fprs = [pair[i][0] for i in range(len(fprs))]
#tprs = [pair[i][1] for i in range(len(tprs))]
#plt.plot(fprs, tprs, color='darkorange', lw=1, label='ROC curve')
#plt.title(r'MTT-bSVD')
#plt.plot(fprs, tprs)
# In[]:
import scipy.io as scio
# In[]:
patient =  '0001448604'
index = 7
CT_df = loadDataFrame('CT')
ROI_df = loadDataFrame('ROI')
#for i, patient in enumerate(patient_list):
img_CT = loadImageArray(CT_df.loc[index, 'Patient_File'], 
                                CT_df.loc[index, 'Series_ID'] )
img_bSVD = loadImage(bSVD_df, 'TTP', patient)
CTP_slices_coord = findSlicesCoordinate(img_bSVD['origin'][2], img_bSVD['spacing'][2], 
                                            img_bSVD['image_array'].shape[0])
roi = loadROI(ROI_df.ROI_path[index])
raw_roi_index = findRoiIndex(roi)

roi_slices_coord = findSlicesCoordinate(img_CT['origin'][2], img_CT['spacing'][2],
                                              img_CT['image_array'].shape[0],
                                              is_roi=True, roi_slices_num=raw_roi_index)
mask = createMask(img_CT['image_array'], roi)
slices_dist = findSlicesDistance(CTP_slices_coord, roi_slices_coord)
roi_index = findClosestIndex(slices_dist)

for i, p in enumerate(roi_index): 
    brain = extractBrainTissue(img_bSVD['image_array'][p])
    labels = brain!=0
#    dst = thresholdDivision(brain, labels, 0.19)
    dst = (brain >= 7)
    r = raw_roi_index[i]
    p1 = plt.subplot(131)
    p2 = plt.subplot(132)
    p3 = plt.subplot(133)
#    lcc = lcc.astype(int)
#    print(np.unique(lcc))
    plt.figure(),
    p1.imshow(brain, 'gray'),
#    p2.imshow(lcc, 'gray'),
    p3.imshow(mask[r], 'gray')
#    plt.imshow(img_CT['image_array'][r - 1],'gray' ,vmin = 35 - 100, vmax = 135 )
#    plt.show()
#    plt.imshow(mask[r], 'gray'),
#    plt.show()
#    scio.savemat('{}.mat'.format(patient + str(r)), {'mask':mask[r]})
        
