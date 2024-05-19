import mediapipe as mp
import screen_brightness_control as sbc
from math import hypot
import numpy as np
import cv2

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode = False,
    max_num_hands = 2,
    min_tracking_confidence = 0.75,
    min_detection_confidence = 0.75,
    model_complexity = 1
)

Draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    Process = hands.process(frameRGB)
    
    landmarkslist = []
    
    if Process.multi_hand_landmarks:
        for handlm in Process.multi_hand_landmarks:
            for _id, landmarks in enumerate(handlm.landmark):
                height, width, color_channels = frame.shape
                
                x, y = int(landmarks.x*width), int(landmarks.y*height)
                landmarkslist.append([id, x, y])
            
            Draw.draw_landmarks(frame, handlm,
                                mpHands.HAND_CONNECTIONS)
            
    if landmarkslist != []:
        x_1,y_1 = landmarkslist[4][1], landmarkslist[4][2]
        x_2,y_2 = landmarkslist[8][1], landmarkslist[8][2]
        
        cv2.circle(frame, (x_1,y_1), 7, (0,255,0), 0)
        cv2.circle(frame, (x_2,y_2), 7, (0,255,0), 0)
        
        cv2.line(frame, (x_1,y_1), (x_2,y_2), (0,255,0), 3)
        
        L = hypot(x_2-x_1, y_2-y_1)
        brightness_level = np.interp(L, [15,220], [0,100])
        
        sbc.set_brightness(brightness_level)
        
        
    
    
    
    cv2.imshow("Screen Brightness Control", frame)
    
    if cv2.waitKey(1) == 27:
        break
    
cv2.destroyAllWindows()