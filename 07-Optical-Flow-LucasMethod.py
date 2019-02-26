import numpy as np
import cv2 ,time
time.sleep(2)
# Parameters for ShiTomasi corner detection (good features to track paper)
corner_track_params = dict(maxCorners = 50,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )


# ### Parameters for Lucas Kanade Optical Flow
# 
# Detect the motion of specific points or the aggregated motion of regions by modifying the winSize argument. This determines the integration window size. Small windows are more sensitive to noise and may miss larger motions. Large windows will “survive” an occlusion.
# 
# The integration appears smoother with the larger window size.
# 
# criteria has two here - the max number (10 above) of iterations and epsilon (0.03 above). More iterations means a more exhaustive search, and a smaller epsilon finishes earlier. These are primarily useful in exchanging speed vs accuracy, but mainly stay the same.
# 
# When maxLevel is 0, it is the same algorithm without using pyramids (ie, calcOpticalFlowLK). Pyramids allow finding optical flow at various resolutions of the image. 

# In[3]:

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (200,200),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10,0.03))


# In[ ]:

# Capture the video
cap = cv2.VideoCapture(0)

# Grab the very first frame of the stream
ret, prev_frame = cap.read()

# Grab a grayscale image (We will refer to this as the previous frame)
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Grabbing the corners
prevPts = cv2.goodFeaturesToTrack(prev_gray, mask = None, **corner_track_params)

# Create a matching mask of the previous frame for drawing on later
mask = np.zeros_like(prev_frame)


while True:
    
    # Grab current frame
    ret,frame = cap.read()
    
    # Grab gray scale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate the Optical Flow on the Gray Scale Frame
    nextPts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, prevPts, None, **lk_params)
    
    # Using the returned status array (the status output)
    # status output status vector (of unsigned chars); each element of the vector is set to 1 if
    # the flow for the corresponding features has been found, otherwise, it is set to 0.
    good_new = nextPts[status==1]
    good_prev = prevPts[status==1]
    
    # Use ravel to get points to draw lines and circles
    for i,(new,prev) in enumerate(zip(good_new,good_prev)):
        
        x_new,y_new = new.ravel()
        x_prev,y_prev = prev.ravel()
        
        # Lines will be drawn using the mask created from the first frame
        mask = cv2.line(mask, (x_new,y_new),(x_prev,y_prev), (0,255,0), 3)
        
        # Draw red circles at corner points
        frame = cv2.circle(frame,(x_new,y_new),8,(0,0,255),-1)
    
    # Display the image along with the mask we drew the line on.
    img = cv2.add(frame,mask)
    cv2.imshow('frame',img)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
   
    # Now update the previous frame and previous points
    prev_gray = frame_gray.copy()
    prevPts = good_new.reshape(-1,1,2)
    
    
cv2.destroyAllWindows()
cap.release()



