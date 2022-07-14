# -*- coding: utf-8 -*-
"""

### To prepare the dataset to train, each image should contain a white letter in a black background, fitted in 64x64 squared box

Firstly, necessary libraries need to be imported as follows
"""

# Commented out IPython magic to ensure Python compatibility.
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

"""Mount Google Drive as follows, because images are stored in Google Drive directories"""

from google.colab import drive
drive.mount('/content/drive')

"""The Pre-processing function as follows"""

def prep(inimg, name):
  imgIni = cv2.cvtColor(inimg, cv2.COLOR_BGR2GRAY)  # Convert the input image to grayscale
  imgIni = cv2.fastNlMeansDenoising(imgIni,None,10,7,21) # Eliminate some noise if having any
  imgIni = cv2.GaussianBlur(imgIni, (3, 3), 0) # Add Gaussian Blur to make the shapes smooth
  ret,img_t = cv2.threshold(imgIni,130,255,cv2.THRESH_BINARY) # Binarize the image

  # To make sure all images are white letters in black background, the following transformations are applied.

  img_arr = np.array(img_t) # Convert image to a 2D array

  row_size = len(img_arr) - 1   # Getting the last index of the row
  col_size = len(img_arr[0]) - 1   # Getting the last index of the column

  tleft = img_arr[0,0]  # Value of the top-left corner pixel
  tright = img_arr[0,col_size]  # Value of the top-right corner pixel
  bleft = img_arr[row_size,0]  # Value of the bottom-left corner pixel
  bright = img_arr[row_size,col_size]  # Value of the bottom-right corner pixel

  # In here, corner pixel values can be 0 (black) or 255 (white) only, because of binary images.
  
  w_cnr = 0   # Set white corner count to 0
  b_cnr = 0   # Set black corner count to 0

  # The following conditions are applied to count the number of white corners and black corners

  if tleft == 255:
    w_cnr += 1
  elif tleft == 0:
    b_cnr += 1

  if tright == 255:
    w_cnr += 1
  elif tright == 0:
    b_cnr += 1

  if bleft == 255:
    w_cnr += 1
  elif bleft == 0:
    b_cnr += 1

  if bright == 255:
    w_cnr += 1
  elif bright == 0:
    b_cnr += 1

  # The following condition is applied to check whether the white corner count is greater than the black corner count. If so, inverse the image applying Bitwise Not operation

  if w_cnr > b_cnr:
    img_t = cv2.bitwise_not(img_t)

  # At this stage, all the images contain white letters in black background 

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

      # To isolate letters, and remove the unwanted black border occured due to manual cropping, the follwoing transformations are applied
      # Referred from: https://newbedev.com/how-to-crop-or-remove-white-background-from-an-image

      kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
      img_mphrd = cv2.morphologyEx(img_blk, cv2.MORPH_CLOSE, kernel)

      contrs2 = cv2.findContours(img_mphrd, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
      cnts2 = sorted(contrs2, key=cv2.contourArea)[-1]

      x,y,w,h = cv2.boundingRect(cnts2)
      img_dst = img_med[y:y+h, x:x+w]
      img_dst = cv2.bitwise_not(img_dst) 

      # Now, images need to be resized in 64x64 squared box, without affecting letter's aspect ratio. To do so, the following transformations are applied
      # Referred from: https://newbedev.com/resize-an-image-without-distortion-opencv

      h, w = img_dst.shape[:2]
      cnr = None if len(img_dst.shape) < 3 else img_dst.shape[2]
      if h == w: return cv2.resize(img_dst, (64, 64), cv2.INTER_AREA)
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

      img_out = cv2.resize(mask, (64, 64), cv2.INTER_AREA)

      # At this stage, all the necessary transformations in pre-processing are completed.

      cv2_imshow(img_out)   # Preview the pre-processed images

      im_dir = "/content/drive/MyDrive/classification_of_inscriptions_periods/step1_preprocessing/images/output/"  # The output directory, which the pre-processed images are need to be saved
      im_name = im_dir+str(i)   # Taking the input image name and concat with the output directory name

      cv2.imwrite(im_name, img_out, [cv2.IMWRITE_JPEG_QUALITY, 100])   # Write the pre-processed images into the output directory


# To remove the existing files in the output directory
files = glob.glob('/content/drive/MyDrive/classification_of_inscriptions_periods/step1_preprocessing/images/output/*')
for f in files:
  os.remove(f)

# To read all the manually segmented images in the input directory and call the pre-processing function for each image
for i in os.listdir('/content/drive/MyDrive/classification_of_inscriptions_periods/step1_preprocessing/images/input/'):
  image = cv2.imread("/content/drive/MyDrive/classification_of_inscriptions_periods/step1_preprocessing/images/input/"+ i)
  prep(image, i)