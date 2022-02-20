import cv2
import numpy as np

cap = cv2.VideoCapture("assets/video.mp4")

prevCircle = None
dist = lambda x1,y1,x2,y2: (x1-x2)**2+(y1-y2)**2

while True:
    ret, frame = cap.read()
    if not ret: 
        break
    
    frame = cv2.flip(frame, 2)
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurFrame = cv2.GaussianBlur(grayFrame, (17,17), 0)

    circles = cv2.HoughCircles(blurFrame, cv2.HOUGH_GRADIENT, 1.2, 100, param1=100, param2=30, minRadius=75, maxRadius=400)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        chosen = None
        for i in circles[0, :]:
            if chosen is None: chosen = i
            if prevCircle is not None:
                if dist(chosen[0],chosen[1],prevCircle[0],prevCircle[1]) <= dist(i[0],i[1],prevCircle[0],prevCircle[1]):
                    chosen = i
        cv2.circle(frame, (chosen[0], chosen[1]), 1, (0,100,100), 3)
        cv2.circle(frame, (chosen[0], chosen[1]), chosen[2], (255,0,255), 3)
        prevCircle = chosen

    cv2.imshow('circles', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()