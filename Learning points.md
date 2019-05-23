## Learning points 

* **load image array from dicom file**

  ```python
  reader = sitk.ImageSeriesReader()
  dicom_names = reader.GetGDCMSeriesFileNames(filepath, seriesID)
  reader.SetFileNames(dicom_names)
  image = reader.Execute()
  image_array = sitk.GetArrayFromImage(image) # z, x, y
  origin = np.array(image.GetOrigin()).reshape(1,3) # x, y, z
  spacing = np.array(image.GetSpacing()).reshape(1,3)
  ```



* **```assert(isinstance(slices_dist, np.ndarray))```**



* **fill polygons with point set contouring**

  ```python
  point_set = np.int32([point_sets[p]])
  cv2.fillConvexPoly(mask[k - 1], point_set, 1)
  ```



* **Numpy**

  * **```img_mask = (image_slice !=  0)```** 

  * **```dst = np.multiply((image_slice >= thresh), labels)```**

  * save/load in npz/npy as dict

    ```python
    roc = {'fprs':fprs,'tprs':tprs}
    np.save(content+'.npz', roc)
    ```

    ```python
    roc = np.load('{}.npz.npy'.format(mode))
    roc = roc.all()
    fprs = roc['fprs']
    tprs = roc['tprs']
    ```

  * **for broadcasting**

    ```CTP_slices_coord * np.ones((num_roi, num_slices))```



* **skimage**

  * **```bw = morphology.opening(bw,morphology.disk(9)) ```** [learn more](<https://www.cnblogs.com/denny402/p/5132677.html>)

  * Otsu`s method: 

    ```python
    # find the threshhold using Otsu's method
    thresh = filters.threshold_otsu(image_slice)
    
    # closing based on Otsu's threshhold
    bw = morphology.closing(image_slice > thresh, morphology.disk(1))
    
     # get labeled
    labels, num = measure.label(bw,connectivity=1, background=0, return_num=True)
    ```

  * largest connected component

    ```python
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
    ```



* **sklearn**
  * **```fpr,tpr,threshold = roc_curve(flat_rois, flat_dsts, pos_label=1)```**
  * **```metrics.auc(fprs, tprs)```**



* **dictionary**:

  * search for the key in terms of  max\min value

    ```python
    max(auc, key=auc.get)
    ```

  * append

    ```python
    auc = {}
    auc.update({mode: metrics.auc(fprs, tprs)})
    ```

    