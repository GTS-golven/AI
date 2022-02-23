import cv2
import numpy as np

cap = cv2.VideoCapture(0)

#Resizing video function
def rescale_frame(frame, percent=100):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

#Function for drawing the circle
prevCircle = None
dist = lambda x1,y1,x2,y2: (x1-x2)**2+(y1-y2)**2

while True:
    ret, frame = cap.read()
    if not ret: 
        break
    
    #Resizing video
    frame75 = rescale_frame(frame, percent=100)

    #Mirroring video/ camera
    frame75 = cv2.flip(frame75, 2)

    #Making video black/ white
    grayFrame = cv2.cvtColor(frame75, cv2.COLOR_BGR2GRAY)

    #Blurring video to remove noise
    blurFrame = cv2.GaussianBlur(grayFrame, (19,19), 0)

    #Circle detection function
    circles = cv2.HoughCircles(blurFrame, cv2.HOUGH_GRADIENT, 1.2, 5, param1=100, param2=30, minRadius=0, maxRadius=300)

    #Finding where to draw the circle on the screen
    if circles is not None:
        circles = np.uint16(np.around(circles))
        chosen = None
        for i in circles[0, :]:
            if chosen is None: chosen = i
            if prevCircle is not None:
                if dist(chosen[0],chosen[1],prevCircle[0],prevCircle[1]) <= dist(i[0],i[1],prevCircle[0],prevCircle[1]):
                    chosen = i
        #Drawing the circle on the screen
        cv2.circle(frame75, (chosen[0], chosen[1]), 1, (0,100,100), 3)
        cv2.circle(frame75, (chosen[0], chosen[1]), chosen[2], (240,120,46), 3)
        prevCircle = chosen

    
    cv2.imshow('circles', frame75)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()