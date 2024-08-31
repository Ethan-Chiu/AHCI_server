import cv2
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
print("start connecting camera")
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
print("camera connected")