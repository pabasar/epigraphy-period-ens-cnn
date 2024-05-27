import tensorflow as tf
print(tf.__version__)
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

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

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import matplotlib.cm as cm

from google.colab import drive
drive.mount('/content/drive')

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
img = cv2.imread("/content/drive/MyDrive/classification_of_inscriptions_periods/step3_prediction/images/insc_images_test/inscription_wise/5_nilagama_rock.jpg")
segment(img)

def prep(inimg, name, max_height, max_width):
  imgIni = cv2.cvtColor(inimg, cv2.COLOR_BGR2GRAY)  # Convert the input image to grayscale
  imgIni = cv2.fastNlMeansDenoising(imgIni, None, 10, 7, 21)  # Eliminate some noise if having any
  imgIni = cv2.GaussianBlur(imgIni, (3, 3), 0)  # Add Gaussian Blur to make the shapes smooth
  ret, img_t = cv2.threshold(imgIni, 130, 255, cv2.THRESH_BINARY)  # Binarize the image

  # At this stage, all the images contain white letters in black background, due to the transformations applied in segment function

  # The following transformations and calculations are necessary to eliminate the distorted images further

  wp_count = np.sum(img_t == 255)  # Take the summation of the white pixels in image
  bp_count = np.sum(img_t == 0)  # Take the summation of the black pixels in image

  bPerc = (bp_count * 100) / (bp_count + wp_count)  # Take the percentage of black pixels in the image
  wPerc = (wp_count * 100) / (bp_count + wp_count)  # Take the percentage of white pixels in the image

  height = inimg.shape[0]  # Taking the height of the image
  width = inimg.shape[1]  # Taking the width of the image

  rat1 = height / width  # Taking the height to width ratio
  rat2 = width / height  # Taking the width to height ratio

  # The following condition is applied to filter the undistorted images only, considering some common features of the distorted binarized images

  if bPerc <= 90 and wPerc <= 90 and (height >= 0.45 * max_height or width >= 0.45 * max_width) and rat1 > 0.2 and rat2 > 0.2:

    # To only remain the largest blob, and eliminate the other unwanted blobs, the following transformations are applied
    # In here the largest blob is the letter shape

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

      kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
      img_mphrd = cv2.morphologyEx(img_blk, cv2.MORPH_CLOSE, kernel)

      contrs2 = cv2.findContours(img_mphrd, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
      cnts2 = sorted(contrs2, key=cv2.contourArea)[-1]

      x,y,w,h = cv2.boundingRect(cnts2)
      img_dst = img_med[y:y+h, x:x+w]
      img_dst = cv2.bitwise_not(img_dst)

      # Now, images need to be resized in 128x128 squared box, without affecting letter's aspect ratio. To do so, the following transformations are applied

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

      cv2.imwrite(im_name, img_out, [cv2.IMWRITE_JPEG_QUALITY, 100])   # Save the pre-processed images into the preprocessed directory

# To remove the existing files in the preprocessed directory
files = glob.glob('/content/drive/MyDrive/classification_of_inscriptions_periods/step3_prediction/images/preprocessed/*')
for f in files:
  os.remove(f)

# To wait until the segmengtaion completed
import time
time.sleep(3)

# To read all the segmented images in the segmented directory and call the pre-processing function for each image
max_height = 0
max_width = 0
for i in os.listdir('/content/drive/MyDrive/classification_of_inscriptions_periods/step3_prediction/images/segmented/'):
    image = cv2.imread("/content/drive/MyDrive/classification_of_inscriptions_periods/step3_prediction/images/segmented/" + i)
    height = image.shape[0]
    width = image.shape[1]
    max_height = max(max_height, height)
    max_width = max(max_width, width)

for i in os.listdir('/content/drive/MyDrive/classification_of_inscriptions_periods/step3_prediction/images/segmented/'):
    image = cv2.imread("/content/drive/MyDrive/classification_of_inscriptions_periods/step3_prediction/images/segmented/" + i)
    prep(image, i, max_height, max_width)

from keras.initializers import glorot_uniform
from keras import backend as K

ensemble_model = keras.models.load_model('/content/drive/MyDrive/classification_of_inscriptions_periods/step2_training/dataset_update_22.06.17/thinning/medial_axis/cnn_single/comparison/ensemble/model/ens_model')

def testimage(inimg):
    img = np.array(inimg)
    img = img / 255.0
    img = img.reshape(1,128,128,3)

    periods = ["early_brahmi","later_brahmi","medieval_sinhala","modern_sinhala","transitional_brahmi"]   
    preds = ensemble_model.predict(img)    # Predicted probabilities
    pred_index = np.argmax(preds)    # Predicted period index
    pred_name = periods[pred_index]    # Predicted period name
    pred_confidence = preds[0][pred_index] * 100    # Predicted confidence score as percentage

    return pred_name, pred_confidence

def final_predict():
    global predict_list
    predict_list = []

    # Calling the above testimage function for all the pre-processed letter images and adding each prediction to a list
    for i in os.listdir('/content/drive/MyDrive/classification_of_inscriptions_periods/step3_prediction/images/preprocessed/'):
        image = cv2.imread("/content/drive/MyDrive/classification_of_inscriptions_periods/step3_prediction/images/preprocessed/"+ i)
        pred_name, _ = testimage(image)    # Get predicted period name
        predict_list += [pred_name]    # List of predictions

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

    period_counts = [eb_cnt, lb_cnt, tr_cnt, mdv_cnt, mdn_cnt]
    total_count = sum(period_counts)

    top_list = period_counts[:]
    top_list.sort(reverse=True)

    output = []

    if top_list[0] == eb_cnt:
        output += [top_list[0]]
        output += ["early_brahmi"]
    elif top_list[0] == lb_cnt:
        output += [top_list[0]]
        output += ["later_brahmi"]
    elif top_list[0] == tr_cnt:
        output += [top_list[0]]
        output += ["transitional_brahmi"]
    elif top_list[0] == mdv_cnt:
        output += [top_list[0]]
        output += ["medieval_sinhala"]
    elif top_list[0] == mdn_cnt:
        output += [top_list[0]]
        output += ["modern_sinhala"]

    percMajor = round(((output[0]*100)/total_count), 2)

    if percMajor == 100:
        print(output[1], ":", percMajor, "%")
    else:
        if percMajor < 85 and top_list[1] > 0:
            if output[1] == "early_brahmi" and top_list[1] == lb_cnt:
                output += [top_list[1]]
                output += ["later_brahmi"]
            elif output[1] == "later_brahmi" and (top_list[1] == eb_cnt or top_list[1] == tr_cnt):
                if top_list[1] == eb_cnt:
                    output += [top_list[1]]
                    output += ["early_brahmi"]
                else:
                    output += [top_list[1]]
                    output += ["transitional_brahmi"]
            elif output[1] == "transitional_brahmi" and (top_list[1] == lb_cnt or top_list[1] == mdv_cnt):
                if top_list[1] == lb_cnt:
                    output += [top_list[1]]
                    output += ["later_brahmi"]
                else:
                    output += [top_list[1]]
                    output += ["medieval_sinhala"]
            elif output[1] == "medieval_sinhala" and (top_list[1] == tr_cnt or top_list[1] == mdn_cnt):
                if top_list[1] == tr_cnt:
                    output += [top_list[1]]
                    output += ["transitional_brahmi"]
                else:
                    output += [top_list[1]]
                    output += ["modern_sinhala"]
            elif output[1] == "modern_sinhala" and top_list[1] == mdv_cnt:
                output += [top_list[1]]
                output += ["medieval_sinhala"]

        if len(output) > 2:
            percMajor = round(((output[0]*100)/total_count), 2)
            percMinor = round(((output[2]*100)/total_count), 2)
            misclassification_count = total_count - output[0] - output[2]
        else:
            percMinor = 0
            misclassification_count = total_count - output[0]

        misclassification_perc = round((misclassification_count * 100) / total_count, 2)

        print(output[1], ":", percMajor, "%")
        if percMinor > 0:
            print(output[3], ":", percMinor, "%")

        if misclassification_perc > 0:
            print("Misclassification Percentage:", misclassification_perc, "%")
            print()

            y = ["Misclassification"]
            x = [misclassification_perc]
            colors = ['red']

            if percMinor > 0:
                y.insert(0, output[3])
                x.insert(0, percMinor)
                colors.insert(0, 'green')

            y.insert(0, output[1])
            x.insert(0, percMajor)
            colors.insert(0, 'green')

            plt.figure(figsize=(5, 1))
            plt.xlim([0, 100])
            plt.barh(y, x, color=colors, height=0.7)
            plt.gca().invert_yaxis()  
            plt.show()
        else:
            print()

            y = []
            x = []
            colors = []

            if percMinor > 0:
                y.insert(0, output[3])
                x.insert(0, percMinor)
                colors.insert(0, 'green')

            y.insert(0, output[1])
            x.insert(0, percMajor)
            colors.insert(0, 'green')

            plt.figure(figsize=(5, 1))
            plt.xlim([0, 100])
            plt.barh(y, x, color=colors, height=0.7)
            plt.gca().invert_yaxis()  
            plt.show()

final_predict()   

