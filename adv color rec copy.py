import cv2
import pandas as pd
import math
import numpy as np

webcam = cv2.VideoCapture(0) # your zeroth webcam
df = pd.read_csv('colordata.csv')

def color_recognition(r,g,b):
    color = 'Unknown'
    names =[]
    index = []
    threshold = 50
    for i in range(len(df)):
        R = df.loc[i, 'r']
        G = df.loc[i, 'g']
        B = df.loc[i, 'b']
        diff = abs(r - R) + abs(g -G) + abs(b-B)
        if diff <= threshold:
            index.append(diff)
            names.append(df.loc[i,'name'])
    
    
    if len(index)>=0:
        result = sorted(index)[0]
        for j in range(len(index)):
            if index[j]==result:
                color = names[j]

    return color

while True:
    ret, frame = webcam.read()
    width = int(webcam.get(3)) #3 width identifier
    height = int(webcam.get(4))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #ret-> can it function False or True, frame-> the image itself
    
    #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    w_c = int(width/2)
    h_c = int(height/2)

    center = rgb[w_c, h_c]#hsv
    r = center[0]
    g = center[1]
    b = center[2]
    #h = center[0]
    #s = center[1]
    #v = center[2]
    

    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(frame, color_recognition(r,g,b), (0,height-10), font, 2, (0,0,0), 1, cv2.LINE_AA)
    
    cv2.circle(frame, (w_c, h_c), 5, (25, 25, 25), 3)
    cv2.imshow('video',frame)
    print(f'r: {r}, g: {g}, b:{b}')
    if cv2.waitKey(1) == ord('q'):
        break #wait till 1 millisecond, the wait key returns ACII so compares ASCII code to ordinal of q ASCII

webcam.release() # break loop
cv2.destroyAllWindows()