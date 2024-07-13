# import the opencv library 
import cv2 
import time 
  
# define a video capture object 
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # , cv2.CAP_DSHOW
set_width = cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
set_height = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 180)
print(set_width, set_height)
frame_rate = cap.set(cv2.CAP_PROP_FPS, 30)
print(frame_rate)

while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = cap.read() 
  
    # Display the resulting frame 
    cv2.imshow('frame', frame) 
      
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    time.sleep(0.1)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

# After the loop release the cap object 
cap.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 