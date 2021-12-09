# ## **BONUS**
# Determine the best words from the image of the tiles given by the user.

# - Import necessary files

# In[29]:

''' All neccesary libraries are imported '''
import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import timeit
from copy import copy, deepcopy


# In[2]:


''' Necessary extra steps to install the pytesseract library '''
#!sudo apt install tesseract-ocr
#!pip install pytesseract
import pytesseract

import scrabble
from PIL import Image, ImageFilter
import sys


# In[30]:


''' Helper function to convert letters to string '''
def listToString(s): 
    str1 = ""    
    for ele in s:
        if(ele=='1'):
          ele='I'
        if(ele=='0'):
          ele='O' 
        str1 += ele  
    return str1 

'''Helper function to crop the image to obtain the alphabets using contours'''
def crop(loc):
    img = cv2.imread(loc)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)[1]
    contours , hierarchy = cv2.findContours(thresh, 1, 2)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if w>280 and h>40:
            break
    cropped = img[y:y+h, x:x+w]
    cv2.imwrite('crop_tiles.jpg',255 - cropped)
    
    return cropped

'''Helper function to obtain the letters with the cropped image as the input'''
def get_letters(loc):
    img = Image.open(loc).convert('L')
    blackwhite = img.point(lambda x: 255 if x < 166 else 0, '1')
    blackwhite.save("tiles_bw.jpg")
    im = Image.open("tiles_bw.jpg")
    smooth_im = im.filter(ImageFilter.SMOOTH_MORE)
    smooth_im = np.asarray(smooth_im)
    kernel = np.ones((3,3),np.uint8)
    smooth_im = cv2.erode(smooth_im,kernel,iterations = 1)
    # plt.imshow(smooth_im,cmap = 'gray')
    # plt.show()
    text = pytesseract.image_to_string(smooth_im, config='-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 7')
    letters = text.lower()
    letter_list = [x for x in letters]
    # print("Letters Are: " + letters)
    return letter_list


# In[38]:


''' Here we run the cell where the input is the rack of tiles and the output is the list of words '''

location = '../data/tiles7/3.jpeg'
location = './images/tiles/' + sys.argv[1]
tiles = cv2.imread(location)
croppedImage = crop(location)
# display([cv2.cvtColor(tiles, cv2.COLOR_BGR2RGB), cv2.cvtColor(croppedImage, cv2.COLOR_BGR2RGB)], ['Original Image', 'Cropped Image'], [0, 0])
current_tile_letters = get_letters('crop_tiles.jpg')
string = listToString(current_tile_letters)

# print('Best words from the given tiles:')
print(scrabble.helper(string.upper()))


