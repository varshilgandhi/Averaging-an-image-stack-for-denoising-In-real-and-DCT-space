# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 11:08:16 2021

@author: abc
"""

"""

Averaging an image stack for denoising in real and DCT (Discreate Cosine Transform) space 


"""

import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.fftpack import dct, idct
from os import listdir

#read folder
image_dir = "noisy_img/"

#read all the image 
filenames = listdir(image_dir)
filenames.sort()

imgs = []
for f in filenames:
    imgs.append((cv2.imread(image_dir + f, 0)).astype(np.float32))

#find height and width of our images    
height, width = imgs[0].shape

#Apply the weighted average to images and corresponding DCT images, respectively.
avg_img = np.zeros([height, width], np.float32)
dct_avg_img = np.zeros([height, width], np.float32)


#read our all images
for i in range(len(imgs)):
    avg_img = cv2.addWeighted(avg_img, i/(i+1.0), imgs[i], 1/(i+1.0), 0)  #Original image
    dct_avg_img = cv2.addWeighted(dct_avg_img, i/(i+1.0),  dct(imgs[i]), 1/(i+1.0), 0) #DCT image
    

#reverse above images
reverse_img = idct(dct_avg_img)

#Let's visualize above images
plt.imsave("00-dct_averaged_img.jpg", reverse_img, cmap="gray")
plt.imsave("00-averaged_img.jpg", avg_img, cmap="gray")

fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(imgs[0], cmap="gray")
ax1.title.set_text('Input Image 1')
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(imgs[1], cmap="gray")
ax2.title.set_text("Input Image 2")

ax3 = fig.add_subplot(2,2,3)
ax3.imshow(avg_img, cmap="gray")
ax3.title.set_text("Average of Images")
ax4 = fig.add_subplot(2,2,4)
ax4.imshow(reverse_img, cmap="gray")
ax4.title.set_text("Image from DCT average")
plt.show()






