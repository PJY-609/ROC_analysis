# -*- coding: utf-8 -*-
"""
Created on Thu May  9 11:50:11 2019

@author: yujue
"""

import pandas as pd
from constant import PATH, path_bSVD, path_ROI
import os
import SimpleITK as sitk
import numpy as np

def loadDataFrame(filename):
    if filename == 'bSVD':
        df = pd.read_csv(os.path.join(PATH+path_bSVD, 'bSVD.csv'))
        df = df.drop_duplicates(subset='Series_ID', keep='first')
        df = df.reset_index()
    elif filename == 'ROI':
        df = pd.read_csv(os.path.join(PATH+path_ROI, 'ROI.csv'))
    elif filename == 'CT':
        df = pd.read_csv(os.path.join(PATH+path_ROI, 'image.csv'))
        df = df.drop_duplicates(subset='Series_ID', keep='first')
        df = df.reset_index()
    return df

def loadContentList(df):
    content_list = df.Series_Description.unique()
    return content_list

def loadPatientList(df):
    patient_list = df.Patient_ID.unique()
    return patient_list


def loadImageArray(filepath, seriesID):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(filepath, seriesID)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    image_array = sitk.GetArrayFromImage(image)
    origin = np.array(image.GetOrigin()).reshape(1,3) # x, y, z
    spacing = np.array(image.GetSpacing()).reshape(1,3)
    img = {'image_array': image_array, 
           'origin': np.squeeze(origin),
           'spacing': np.squeeze(spacing)}
    return img

    
def loadROI(ROI_path):
    data = np.load(ROI_path)
    roi = data['arr_0'].all()
    return roi

def findRoiIndex(roi):
    roi_index=[]
    for i in roi.keys():
        if roi[i]:
            roi_index.append(i)
    return roi_index

def loadImage(df, content, patient):
    df_temp = df[(df.Series_Description==content) & (df.Patient_ID==patient)].reset_index()
    img = loadImageArray(df_temp.loc[0, 'Patient_File'], df_temp.loc[0, 'Series_ID'])
    return img


    
    
    