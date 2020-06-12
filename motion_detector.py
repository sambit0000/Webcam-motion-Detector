# Importing the required libraries
import cv2
from datetime import datetime
import pandas

# Switching on the webcam
video = cv2.VideoCapture(0)

# Creating empty variable or list
first_frame = None
status_list = [None, None]
times = []
# Creating empty dataframe
df = pandas.DataFrame(columns = ["Start","End"])

# Setting up the infinite loop
while True:

    # Capturing frame from webcam
    check, frame = video.read()
    # Status if there is no motion
    status = 0
    # Converting frame to gray image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Reducing noise in gray image
    gray = cv2.GaussianBlur(gray,(21,21),0)
    # Fixing the first frame
    if first_frame is None:
        first_frame = gray
        continue
    # Difference between first frame and subsequent frames
    delta_frame = cv2.absdiff(first_frame,gray)
    # To get threshold frame above certain threshold of intensity
    thresh_frame = cv2.threshold(delta_frame,50,255,cv2.THRESH_BINARY)[1]
    # Smoothening the frame
    thresh_frame = cv2.dilate(thresh_frame,None,iterations=2)
    # Finding contours on the images in the frame
    (cnts,_) = cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        # Selecting those images above certain pixels
        if cv2.contourArea(contour) < 3500:
            continue
        # Status if there is motion
        status = 1
        # Finding coordinates of the object bounded by contour
        (x,y,w,h) = cv2.boundingRect(contour)
        # Drawing rectangle to identify the moving object
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 3)
    # Get the time log when the object comes and goes
    status_list.append(status)
    if status_list[-1]==1 and status_list[-2]==0:
        times.append(datetime.now())
    elif status_list[-1]==0 and status_list[-2]==1:
        times.append(datetime.now())
    # Window showing all the frames  
    cv2.imshow("Gray Frame",gray)
    cv2.imshow("Delta Frame",delta_frame)
    cv2.imshow("Threshold Frame",thresh_frame)
    cv2.imshow("color frame", frame)
    # Display frame after every 1 millisecond
    key = cv2.waitKey(1)
    # Quit the loop when 'q' is pressed
    if key==ord('q'):
        if status==1:
            times.append(datetime.now())
        break

print(status_list)
print(times)
# Get the start and end time of object into a dataframe
for i in range(0,len(times),2):
    df = df.append({"Start":times[i],"End":times[i+1]},ignore_index=True)
# Write the df into a csv file
df.to_csv("Times.csv")
# Release the webcam
video.release()

cv2.destroyAllWindows()
