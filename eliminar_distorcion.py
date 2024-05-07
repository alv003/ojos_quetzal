import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt

# Load camera matrix and distortion coefficients
mtx = np.array([[9.026271670864188081e+02,0.000000000000000000e+00,6.638151779030837361e+02],[0.000000000000000000e+00,9.147711304614931578e+02,4.301660543312769391e+02], [0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00]])
dist = np.array([-2.234676752467908978e-01,7.475428582944396716e-01,1.052375814961837956e-02,4.051515941641855960e-03,-1.018166970395988580e+00])

#Remove distortion
img = cv2.imread('camera_01\\WIN_20240506_14_08_57_Pro.jpg')
if img is None:
    print('Failed to load image')
else:
    h,  w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite('calibresult_1.png',dst)
    cv2.imshow('calibresult_1.png',dst)
    cv2.waitKey(0)
