import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self,mode=False,maxHand=2,detectionConf=0.5,trackConf=0.5):
        self.mode=mode
        self.maxHand=maxHand
        self.detectionConf=detectionConf
        self.trackConf=trackConf

        self.mpHands=mp.solutions.hands
        self.hands=self.mpHands.Hands(self.mode,self.maxHand,
                        self.detectionConf,self.trackConf)
        self.mpDraw=mp.solutions.drawing_utils
        self.tipID=[4,8,12,16,20]

    def findHand(self,img,draw=True):

        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if(self.results.multi_hand_landmarks):
            for handLns in self.results.multi_hand_landmarks:
                if(draw):
                    self.mpDraw.draw_landmarks(img,handLns,self.mpHands.HAND_CONNECTIONS)
                else:
                    self.mpDraw.draw_landmarks(img,handLns)
        
        return img
    
    def findPosition(self,img,handNo=0,draw=True):

        self.lmList=[]

        if(self.results.multi_hand_landmarks):

            myHand=self.results.multi_hand_landmarks[handNo]

            for id,lm in enumerate(myHand.landmark):
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                self.lmList.append([id,cx,cy])
                if(draw):
                    cv2.circle(img,(cx,cy),3,(255,0,0),cv2.FILLED)

        return self.lmList

    def fingersUp(self):
        fingers=[]
        if(len(self.lmList)!=0):

            if(self.lmList[17][1]>self.lmList[4][1]):
                if(self.lmList[4][1]<self.lmList[2][1]+10):
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if(self.lmList[4][1]>self.lmList[2][1]+10):
                    fingers.append(1)
                else:
                    fingers.append(0)

            for id in range(1,5):
                if(self.lmList[self.tipID[id]][2]<self.lmList[self.tipID[id]-2][2]):
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

def main():
    cTime=0
    pTime=0
    cap=cv2.VideoCapture(0)
    detector=handDetector()

    while True:
        success,img=cap.read()

        img=detector.findHand(img,True)
        lmList=detector.findPosition(img)

        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime

        cv2.putText(img,str((fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

        cv2.imshow("Image",img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()