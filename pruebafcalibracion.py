import cv2
import numpy as np
import os.path
import sys
import matplotlib.pyplot as plt
from picamera2 import Picamera2, Preview
import time


c= np.array([[ 9.81552078e-01, -5.25641775e-03,  5.36823210e+00], [-9.75914258e-03,  9.87968755e-01,  8.78222247e+00], [-1.97827135e-05, -6.43749669e-06,  1.00000000e+00]])


# Create Sift object :3
sift = cv2.SIFT_create()

# radio, es el tamaño de la lente en centimetros
class Correccion: 
            
    def img_alignment_sequoia(self, img_base_NIR, img_RED, width, height):    
            global c

            # Resize the images to the same size specified in image_SIZE
            base_NIR = cv2.resize(img_base_NIR, (width, height), interpolation=cv2.INTER_LINEAR)
            b_RED = cv2.resize(img_RED, (width, height), interpolation=cv2.INTER_LINEAR)

            # Stabilize the RED image with respect to the NIR base image
            stb_RED =cv2.warpPerspective(b_RED,c,(width, height))

            return base_NIR, stb_RED


correccion_img = Correccion()
#define image size
width = 700
height = 500

# Reading images
Img_RED = cv2.imread(r'/home/kouriakova/Ojos de Quetzal/ojos_quetzal/IMG_700101_001240_0000_RED.TIF',0)
Img_NIR = cv2.imread(r'/home/kouriakova/Ojos de Quetzal/ojos_quetzal/IMG_700101_001240_0000_NIR.TIF',0)
# Create a BGR image with the red and nir
merged_fix_bad = cv2.merge((Img_RED,Img_RED,Img_NIR)) # First image, misaligned
merged_fix_bad = cv2.resize(merged_fix_bad, (700, 500), interpolation=cv2.INTER_LINEAR)
# Assuming merged_fix_bad is supposed to be an RGB image


stb_NIR, stb_RED =  correccion_img.img_alignment_sequoia(Img_NIR, Img_RED, width, height)
merged_fix_stb = cv2.merge((stb_RED,stb_RED, stb_NIR))    

 # Use matplotlib to show the images
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(merged_fix_bad)
plt.subplot(1, 2, 2)
plt.imshow(merged_fix_stb)


plt.title('Aligned image')
plt.show()

print()

