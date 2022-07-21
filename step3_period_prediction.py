# -*- coding: utf-8 -*-
"""step3_period_prediction.ipynb

### Final Prediction. To support decision making.

Firstly, necessary libraries need to be imported as follows

---
"""

import numpy as np
import pandas as pd
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

from PIL import Image
from keras.preprocessing import image

import cv2
import os 
import glob
import argparse

from google.colab.patches import cv2_imshow

import tensorflow as tf
from tensorflow import keras

"""Mount Google Drive as follows, because images and trained model are stored in Google Drive directories"""

from google.colab import drive
drive.mount('/content/drive')

"""Segmenting function as follows"""

def segment(inimg):
  imgCpy = inimg.copy()    # Make a copy of the image 
  imgIni = cv2.cvtColor(inimg, cv2.COLOR_BGR2GRAY)    # Convert the input image to grayscale
  imgIni = cv2.fastNlMeansDenoising(imgIni,None,10,7,21)    # Eliminate some noise if having any
  imgIni = cv2.GaussianBlur(imgIni, (3, 3), 0)    # Add Gaussian Blur to make the shapes smooth
  ret,img_t = cv2.threshold(imgIni,130,255,cv2.THRESH_BINARY)   # Binarize the image

  # To make sure all images are white letters in black background, the following transformations are applied.

  wp = np.sum(img_t == 255)   # Taking summation of white pixels in image
  bp = np.sum(img_t == 0)   # Taking summation of black pixels in image

  # The following condition is to inverse image, if the summation of white pixels is greater than black pixels

  if wp > bp:
    img_t = cv2.bitwise_not(img_t)

  # At this stage, all the images contain white letters in black background

  # The following transformations are applied to isolate and crop each white blob in the image and save them in a different directory
  # Referred from: https://newbedev.com/extract-all-bounding-boxes-using-opencv-python

  contrs = cv2.findContours(img_t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contrs = contrs[0] if len(contrs) == 2 else contrs[1]

  seg_num = 0
  for i in contrs:
    x,y,w,h = cv2.boundingRect(i)
    seg = img_t[y:y+h, x:x+w]
    cv2.imwrite('/content/drive/MyDrive/classification_of_inscriptions_periods/step3_prediction/images/segmented/seg_{}.jpg'.format(seg_num), seg)
    cv2.rectangle(img_t,(x,y),(x+w,y+h),(36,255,12),2)
    seg_num += 1

  # Now the segmentation is completed. But in addition to letters, unwanted white blobs also segmented. They are eliminated in the next pre-processing function

  cv2_imshow(inimg)   # Preview input image, a part of estampage containing a few letters

# To remove the existing files in the segmented directory
files = glob.glob('/content/drive/MyDrive/classification_of_inscriptions_periods/step3_prediction/images/segmented/*')
for f in files:
  os.remove(f)

# To read the input image, a part of estampage containing a few letters and call the segment function
img = cv2.imread("/content/drive/MyDrive/classification_of_inscriptions_periods/step3_prediction/images/insc_images_test/inscription_wise/eb11.jpg")
segment(img)

"""The Pre-processing function as follows. The function is a bit similar to the pre-processing function in first step. But in here, output images are letters fitted in 128x128 squared boxes, since the input size of the trained model is 128x128"""

def prep(inimg, name):
  imgIni = cv2.cvtColor(inimg, cv2.COLOR_BGR2GRAY)  # Convert the input image to grayscale
  imgIni = cv2.fastNlMeansDenoising(imgIni,None,10,7,21) # Eliminate some noise if having any
  imgIni = cv2.GaussianBlur(imgIni, (3, 3), 0) # Add Gaussian Blur to make the shapes smooth
  ret,img_t = cv2.threshold(imgIni,130,255,cv2.THRESH_BINARY) # Binarize the image

  # At this stage, all the images contain white letters in black background, due to the transformations applied in segment function

  # The following transformations and calculations are necessary to eliminate the distorted images further

  wp_count = np.sum(img_t == 255)   # Take the summation of the white pixels in image
  bp_count = np.sum(img_t == 0)   # Take the summation of the black pixels in image

  bPerc = (bp_count * 100) / (bp_count + wp_count)    # Take the percentage of black pixels in the image
  wPerc = (wp_count * 100) / (bp_count + wp_count)    # Take the percentage of white pixels in the image

  height = inimg.shape[0]   # Taking the height of the image
  width = inimg.shape[1]    # Taking the width of the image

  rat1 = height / width   # Taking the height to width ratio
  rat2 = width / height   # Taking the width to height ratio

  # The following condition is applied to filter the undistorted images only, considering some common features of the distorted binarized images

  if bPerc <= 90 and wPerc <= 90 and height >= 15 and width >= 15 and rat1 > 0.2 and rat2 > 0.2:

    # To only remain the largest blob, and eliminate the other unwanted blobs, the following transformations are applied
    # In here the largest blob is the letter shape
    # Referred from: https://www.javaer101.com/en/article/34980509.html

    temp = cv2.morphologyEx(img_t, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1)))
    contrs, _ = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    w1 = np.sum(img_t == 255)
    b1 = np.sum(img_t == 0)   

    # In some cases, the processed images upto now can be totally white or black, To eliminate these kinds of images the following condition is applied

    if w1 != 0 and b1 != 0:
      cnts = max(contrs, key=cv2.contourArea)
      img_blk = np.zeros(img_t.shape, np.uint8)
      cv2.drawContours(img_blk, [cnts], -1, 255, cv2.FILLED)
      img_blk = cv2.bitwise_and(img_t, img_blk)
      img_shp = cv2.bitwise_not(img_blk) 
      img_med = cv2.cvtColor(img_shp, cv2.COLOR_BGR2RGB)

      # At this stage, the image only consists of the white letter in black background, without unwanted blobs

      # To isolate letters, and remove the unwanted black border occured due to the segmentation, the follwoing transformations are applied
      # Referred from: https://newbedev.com/how-to-crop-or-remove-white-background-from-an-image

      kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
      img_mphrd = cv2.morphologyEx(img_blk, cv2.MORPH_CLOSE, kernel)

      contrs2 = cv2.findContours(img_mphrd, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
      cnts2 = sorted(contrs2, key=cv2.contourArea)[-1]

      x,y,w,h = cv2.boundingRect(cnts2)
      img_dst = img_med[y:y+h, x:x+w]
      img_dst = cv2.bitwise_not(img_dst) 

      # Now, images need to be resized in 128x128 squared box, without affecting letter's aspect ratio. To do so, the following transformations are applied
      # Referred from: https://newbedev.com/resize-an-image-without-distortion-opencv

      h, w = img_dst.shape[:2]
      cnr = None if len(img_dst.shape) < 3 else img_dst.shape[2]
      if h == w: return cv2.resize(img_dst, (128, 128), cv2.INTER_AREA)
      if h > w: dif = h
      else:     dif = w
      x_pos = int((dif - w)/2.)
      y_pos = int((dif - h)/2.)
      if cnr is None:
        mask = np.zeros((dif, dif), dtype=img_dst.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img_dst[:h, :w]
      else:
        mask = np.zeros((dif, dif, cnr), dtype=img_dst.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img_dst[:h, :w, :]

      img_out = cv2.resize(mask, (128, 128), cv2.INTER_AREA)

      # At this stage, all the necessary transformations in pre-processing are completed.

      cv2_imshow(img_out)   # Preview the pre-processed images

      im_dir = "/content/drive/MyDrive/classification_of_inscriptions_periods/step3_prediction/images/preprocessed/"  # The output directory, which the pre-processed images are need to be saved
      im_name = im_dir+str(i)   # Taking the input image name and concat with the output directory name

      cv2.imwrite(im_name, img_out, [cv2.IMWRITE_JPEG_QUALITY, 100])   # Write the pre-processed images into the preprocessed directory


# To remove the existing files in the preprocessed directory
files = glob.glob('/content/drive/MyDrive/classification_of_inscriptions_periods/step3_prediction/images/preprocessed/*')
for f in files:
  os.remove(f)

# To wait until the segmengtaion completed
import time
time.sleep(3)

# To read all the segmented images in the segmented directory and call the pre-processing function for each image
for i in os.listdir('/content/drive/MyDrive/classification_of_inscriptions_periods/step3_prediction/images/segmented/'):
  image = cv2.imread("/content/drive/MyDrive/classification_of_inscriptions_periods/step3_prediction/images/segmented/"+ i)
  prep(image, i)

"""Loading the model, trained in the second step"""

from keras.initializers import glorot_uniform
from keras import backend as K

#model = keras.models.load_model('/content/drive/MyDrive/classification_of_inscriptions_periods/step2_training/dataset_update_22.06.17/model/mymodel')
model = keras.models.load_model('/content/drive/MyDrive/classification_of_inscriptions_periods/step2_training/dataset_update_22.06.17/model/weights_best_VGGnew.hdf5')
#model.summary()

"""Function to prepare each letter images and return predicted period of each letter, based on the trained model"""

def testimage(inimg):
  img = np.array(inimg)
  img = img / 255.0
  img = img.reshape(1,128,128,3)

  periods = ["early_brahmi","later_brahmi","medieval_sinhala","modern_sinhala","transitional_brahmi"]   # List of periods
  pred_name = periods[np.argmax(model.predict(img))]    # Predicted period, from the periods available on the list

  return pred_name

"""Prediction Function to give final output, the predicted period, considering features of each letter"""

def final_predict():

  global predict_list
  predict_list = []

  # Calling the above testimage function for all the pre-processed letter images and adding each prediction to a list
  for i in os.listdir('/content/drive/MyDrive/classification_of_inscriptions_periods/step3_prediction/images/preprocessed/'):
    image = cv2.imread("/content/drive/MyDrive/classification_of_inscriptions_periods/step3_prediction/images/preprocessed/"+ i)
    predict_list += [testimage(image)]    # List of predictions 

  # Initializing period count as 0  
  eb_cnt = 0
  lb_cnt = 0
  tr_cnt = 0
  mdv_cnt = 0
  mdn_cnt = 0

  # Counting number of each period's predictions
  for i in range(len(predict_list)):
    if predict_list[i] == "early_brahmi":
      eb_cnt += 1
    if predict_list[i] == "later_brahmi":
      lb_cnt += 1
    if predict_list[i] == "transitional_brahmi":
      tr_cnt += 1
    if predict_list[i] == "medieval_sinhala":
      mdv_cnt += 1
    if predict_list[i] == "modern_sinhala":
      mdn_cnt += 1

  top_list = [eb_cnt, lb_cnt, tr_cnt, mdv_cnt, mdn_cnt]
  top_list.sort(reverse=True)

  output = []

  if top_list[1] == 0:

    if top_list[0] == eb_cnt:
      print("Early Brahmi Period")
    elif top_list[0] == lb_cnt:
      print("Later Brahmi Period")
    elif top_list[0] == tr_cnt:
      print("Transitional Brahmi Period")
    elif top_list[0] == mdv_cnt:
      print("Medieval Sinhala Period")
    elif top_list[0] == mdn_cnt:
      print("Modern Sinhala Period")

  else:

    for i in range(2):
      if top_list[i] > 0:
        if top_list[i] == eb_cnt:
          output += [top_list[i]]
          output += ["Early Brahmi Period"]
        elif top_list[i] == lb_cnt:
          output += [top_list[i]]
          output += ["Later Brahmi Period"]
        elif top_list[i] == tr_cnt:
          output += [top_list[i]]
          output += ["Transitional Brahmi Period"]
        elif top_list[i] == mdv_cnt:
          output += [top_list[i]]
          output += ["Medieval Sinhala Period"]
        elif top_list[i] == mdn_cnt:
          output += [top_list[i]]
          output += ["Modern Sinhala Period"]

      if len(output) > 2:

      # Verify the second period is consecutive. If so, print the percentages
        if (output[1] == "Early Brahmi Period" and output[3] == "Later Brahmi Period") or (output[1] == "Later Brahmi Period" and (output[3] == "Early Brahmi Period" or output[3] == "Transitional Brahmi Period")) or (output[1] == "Transitional Brahmi Period" and (output[3] == "Later Brahmi Period" or output[3] == "Medieval Sinhala Period")) or (output[1] == "Medieval Sinhala Period" and (output[3] == "Transitional Brahmi Period" or output[3] == "Modern Sinhala Period")) or (output[1] == "Modern Sinhala Period" and output[3] == "Medieval Sinhala Period"):
        
          percMajor = round(((output[0]*100)/(output[0]+output[2])), 2)
          percMinor = round(((output[2]*100)/(output[0]+output[2])), 2)

          if percMajor < 85:

            print(percMajor, "% of letters :", output[1])
            print(percMinor, "% of letters :", output[3])

            print()

            y = [output[3], output[1]]
            x = [percMinor, percMajor]

            plt.figure(figsize=(5,1))
            plt.xlim([0, 100])
            plt.barh(y, x, height=0.7)
            #plt.xlabel("Percentage")
            plt.show()

          else:
            print(output[1])

        else:
          print(output[1])

      
final_predict()   # Calling the prediction function

