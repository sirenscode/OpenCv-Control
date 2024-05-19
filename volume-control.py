import cv2
import linearwinvolume
import mediapipe as mp
from math import hypot
import numpy as np

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    min_tracking_confidence = 0.75,
    max_num_hands = 2,
    model_complexity = 1,
    min_detection_confidence = 0.75
)

cap = cv2.VideoCapture(0)

Draw = mp.solutions.drawing_utils

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    height, width, _ = frame.shape
    
    landmarkslist = []
    
    Process = hands.process(frame)
    if Process.multi_hand_landmarks:
        for handlm in Process.multi_hand_landmarks:
            for _, landmarks in enumerate(handlm.landmark):
                x, y = int(landmarks.x*width), int(landmarks.y*height)
                landmarkslist.append([id,x,y])
            
            Draw.draw_landmarks(frame, handlm,
                                mpHands.HAND_CONNECTIONS)
    
    if landmarkslist != []:
        x_1, y_1 = landmarkslist[4][1], landmarkslist[4][2]
        x_2, y_2 = landmarkslist[8][1], landmarkslist[8][2]
        
        cv2.circle(frame, (x_1,y_1), 7, (0,255,0), 3)
        cv2.circle(frame, (x_2,y_2), 7, (0,255,0), 3)
        
        cv2.line(frame, (x_1,y_1), (x_2,y_2), (0,255,0), 3)
        
        L = hypot(x_2-x_1, y_2-y_1)
        
        volume_level = np.interp(L, [15,220], [0,100])
        
        linearwinvolume.set_volume(volume_level+2)
        
        
        
            
    cv2.imshow("Volume Control", frame)
    
            
    
    if cv2.waitKey(1) == 27:
        break
    
cv2.destroyAllWindows()