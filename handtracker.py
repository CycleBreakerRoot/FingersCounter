import mediapipe as mp
import cv2 as cv
import time
import math

###################
###################
class HandsDetector():
    
    def __init__(self , mode = False , MaxHands = 2 , DetectionCon = 0.5 , TrackCon = 0.5):
    
        self.mode = mode
        self.MaxHands = MaxHands
        self.DetectionCon = DetectionCon
        self.TrackCon = TrackCon


        self.mpHands = mp.solutions.mediapipe.python.solutions.hands
        self.hands = self.mpHands.Hands(self.mode ,self.MaxHands,1,self.DetectionCon ,self.TrackCon )
        self.mpDraw = mp.solutions.mediapipe.python.solutions.drawing_utils
        
    
    def FindHands(self , img , draw = True):


        imgRGB = cv.cvtColor(img , cv.COLOR_BGR2RGB)
        self.res = self.hands.process(imgRGB)
        
        if self.res.multi_hand_landmarks:
            for handLms in self.res.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)
                    
    
    def FindPosition(self, img , HandNo = 0 , draw = True):
        
        LmList = []
        
        if self.res.multi_hand_landmarks:
            myHand = self.res.multi_hand_landmarks[HandNo]
            for id , lm in enumerate(myHand.landmark):
                LmList.append([id , lm.x , lm.y])
        return LmList
    
    
    def FindDistance(self , mylist , n , m):
        if len(mylist) != 0:
            x_n , y_n = mylist[n][1] , mylist[n][2]
            x_m , y_m = mylist[m][1] , mylist[m][2]
            
            return math.sqrt((x_n - x_m) ** 2 + (y_n - y_m) ** 2)
        return 0
    
                
    



def main():
    print('hello')   
    detector = HandsDetector()
    pTime =0
    cTime = 0

    cap = cv.VideoCapture(0)

    while True:
        scs , img = cap.read()
        img = cv.resize(img , (1000 ,800))
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv.putText(img , str(int(fps)) , (10 , 70) ,  cv.FONT_HERSHEY_PLAIN,3,(255,0,255) , 3)
        
        detector.FindHands(img) 
        l = detector.FindPosition(img , draw= False)
        print(detector.FindDistance(l , 4, 8))
        
        cv.imshow('s' , img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break    



if __name__ == "__main__":
    main()