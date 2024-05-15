import cv2
from picamera2 import Picamera2, Preview
import time
import os

cam0=Picamera2(0)
cam1=Picamera2(1)
cam_config = cam0.create_still_configuration(main={"size":(1920, 1080)}, lores={"size":(700, 500)}, display ="lores")
cam_config1 = cam1.create_still_configuration(main={"size":(1920, 1080)}, lores={"size":(700, 500)}, display ="lores")

cam0.configure(cam_config)
cam0.start_preview(Preview.QTGL)

cam1.configure(cam_config1)
cam1.start_preview(Preview.QTGL)

cam0.start_and_capture_files("test{:d}.jpg",initial_delay=0.1, delay=0.1, num_files=1)
cam1.start_and_capture_files("test1{:d}.jpg",initial_delay=0.1, delay=0.1, num_files=1)

