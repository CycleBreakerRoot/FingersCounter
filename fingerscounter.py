import handtracker
import cv2 as cv
import time
import math

cap = cv.VideoCapture(0)
detector = handtracker.HandsDetector(MaxHands=1)


def IsTheFingerPointingUp(l , r , f):
    x_r , y_r= l[r][1] , l[r][2]
    x_f , y_f= l[f][1] , l[f][2]
    
    wrist_x , wrist_y = l[0][1] , l[0][2]
    
    wrist_r_mag = math.sqrt((x_r - wrist_x) **2 + (y_r - wrist_y) ** 2)
    f_wrist_mag = math.sqrt((wrist_x - x_f) **2 + (wrist_y - y_f) ** 2)
    
    return  f_wrist_mag >= wrist_r_mag
 
def IsTheThumbPointingUp(l , r , f):
    x_r , y_r= l[r][1] , l[r][2]
    x_f , y_f= l[f][1] , l[f][2]
    
    wrist_x , wrist_y = l[5][1] , l[5][2]
    
    wrist_r_mag = math.sqrt((x_r - wrist_x) **2 + (y_r - wrist_y) ** 2)
    f_wrist_mag = math.sqrt((wrist_x - x_f) **2 + (wrist_y - y_f) ** 2)
    
    return  f_wrist_mag >= wrist_r_mag       

pTime = 0
while True:
    scs , img = cap.read()
    img = cv.resize(img , (1200 , 900))
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    n = 0
    cv.putText(img , str(int(fps)) , (10 , 70) ,  cv.FONT_HERSHEY_PLAIN,3,(255,0,255) , 3)
    detector.FindHands(img)
    l = detector.FindPosition(img , draw= False)
    if len(l) != 0:
        Index = IsTheFingerPointingUp(l , 5 , 8)
        midle = IsTheFingerPointingUp(l , 9 , 12)
        ring = IsTheFingerPointingUp(l , 13 , 16)
        pinky = IsTheFingerPointingUp(l , 17 , 20)
        thumb = IsTheThumbPointingUp(l , 3 , 4)

        n = int(Index) + int(midle) + int(ring) + int(pinky) + int(thumb)
    else:
        n = 0
        
    cv.putText(img , str(int(n)) , (1000 , 70) ,  cv.FONT_HERSHEY_PLAIN,3,(255,0,255) , 3)
    cv.imshow(' ' , img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break    