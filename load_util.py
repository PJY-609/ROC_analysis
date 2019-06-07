# -*- coding: utf-8 -*-
"""
Created on Thu May  9 11:50:11 2019

@author: yujue
"""

import pandas as pd
from constant import PATH
import os
import SimpleITK as sitk
import numpy as np

def loadDataFrame(filetype):
    """Load dataframe from local .csv file

    Args:
        filetype: expecting 'ROI', 'bSVD', 'sSVD', 'CT'

    Returns:
        a dataframe
    """
    df = pd.read_csv(os.path.join(PATH['base'] + PATH[filetype], 
                                  '{}.csv'.format(filetype)))
    if filetype != 'ROI':
        df = df.drop_duplicates(subset='Series_ID', keep='first')
        df = df.reset_index()

    return df

def loadContentList(df):
    """Acquire all the types of CTP parameters containing in the dataframe

    Args:
        dataframe

    Returns:
        a list: 
            e.g. ['MTT-bSVD', 'TTP', 'CBV-bSVD']
    """
    content_list = df.Series_Description.unique()
    return content_list

def loadPatientList(df):
    """Acquire all patient ID from dataframe

    Args:
        dataframe

    Returns:
        a list
    """
    patient_list = df.Patient_ID.unique()
    return patient_list


def loadImageArray(filepath, seriesID):
    """Retrieve the image array, image origin, and image spacing 
    based on the input filepath and seriesID. SeriesID determines 
    which content the returning dict refers to

    Args:
        filepath: the local path of dicom files
        seriesID: extracted from dataframe

    Returns:
        a dict contains 
        {'images_array': numpy.ndarray  
         'origin': real world coordinate of the origin of image_array
         'spacing': a list contains the spacing of each pixel in 
                    three dimensions}
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(filepath, seriesID)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    image_array = sitk.GetArrayFromImage(image) # z, x, y
    origin = np.array(image.GetOrigin()).reshape(1,3) # x, y, z
    spacing = np.array(image.GetSpacing()).reshape(1,3)
    img = {'image_array': image_array, 
           'origin': np.squeeze(origin),
           'spacing': np.squeeze(spacing)}
    return img

    
def loadROI(ROI_path):
    """load roi from local .npz file

    Args:
        ROI_path: local path of .npz file

    Returns:
        a dict: 
            the keys are the order of slice with ROI 
            the values are point sets of ROI
    """
    data = np.load(ROI_path)
    roi = data['arr_0'].all()
    
    return roi

def findRoiIndex(roi):
    """Get rid of the some slices with empty point set

    Args:
        roi: dict

    Returns:
        a list:the order of slices with non-empty point set 
    """
    roi_index=[]
    for i in roi.keys():
        if roi[i]:
            i -= 1 # roi中存储的是切片层数 需要减一
            roi_index.append(i)
    return roi_index

def loadImage(df, content, patient):
    """Get a dict of image_array based on content and patient

    Args:
        df: dataframe
        content: for example, 'MTT-bSVD'
        patient: patient ID, for example, 'NCT68331'

    Returns:
        a dictionary of image_array
    """
    df_temp = df[(df.Series_Description==content) & (df.Patient_ID==patient)].reset_index()
    img = loadImageArray(df_temp.loc[0, 'Patient_File'], df_temp.loc[0, 'Series_ID'])
    return img


    

    
    
    