import cv2
import numpy as np
import time
import os
import hand_traking_module as htm

folderPath="header"
myList=os.listdir(folderPath)
# print(myList)

overlayList=[]

for path in myList:
    image=cv2.imread(f'{folderPath}/{path}')
    overlayList.append(image)

# print(overlayList)

cap=cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)


header=overlayList[0]
drawColor=(0,0,0)
brushThickness=15
eraserThickness=50
xp,yp=0,0
imgCanvas=np.zeros((720,1280,3),np.uint8)

detector=htm.handDetector(maxHand=1,detectionConf=0.75)

while(True):
    success,img=cap.read()
    img=cv2.flip(img,1)

    img=detector.findHand(img)
    lmList=detector.findPosition(img,draw=False)

    if(len(lmList)!=0):
        x1,y1=lmList[8][1:]
        x2,y2=lmList[12][1:]

        fingers=detector.fingersUp()

        if(fingers[1] and fingers[2]):
            if(y1<150):
                if(250<x1<450):
                    header=overlayList[1]
                    drawColor=(0,0,255)
                elif(500<x1<750):
                    header=overlayList[2]
                    drawColor=(255,0,0)
                elif(800<x1<950):
                    header=overlayList[3]
                    drawColor=(0,255,0)
                elif(1000<x1<1200):
                    drawColor=(0,0,0)
                    header=overlayList[4]

            xp,yp=x1,y1
            cv2.rectangle(img,(x1,y1-25),(x2,y2+25),drawColor,cv2.FILLED)


        if(fingers[1] and fingers[2]==False):
            cv2.circle(img,(x1,y1),15,drawColor,cv2.FILLED)

            if(xp==0 and yp==0):
                xp,yp=x1,y1

            if(drawColor==(0,0,0)):
                cv2.line(img,(xp,yp),(x1,y1),drawColor,eraserThickness)
                cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,eraserThickness)
            else:
                cv2.line(img,(xp,yp),(x1,y1),drawColor,brushThickness)
                cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,brushThickness)
            xp,yp=x1,y1

    re=cv2.resize(header,(1280,150),interpolation=cv2.INTER_AREA)
    img[0:150,0:1280]=re


    imgGrey=cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
    _, imgInv=cv2.threshold(imgGrey,50,255,cv2.THRESH_BINARY_INV)
    imgInv=cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img=cv2.bitwise_and(img,imgInv)
    img=cv2.bitwise_or(img,imgCanvas)

    # img=cv2.addWeighted(img,0.5,imgCanvas,0.5,0)

    cv2.imshow("Image",img)
    # cv2.imshow("Canvas",imgCanvas)
    if(cv2.waitKey(1) & 0xFF==ord('q')):
        break

cap.release()
cv2.destroyAllWindows()