import cv2 as cv 
import numpy as np 
import os 
import matplotlib.pyplot as plt 


def mask(picture):
    height,width=picture.shape
    triangle = np.array([[(80, height), (530, 230), (width, height)]], dtype=np.int32)
    mask=np.zeros_like(picture)
    mask=cv.fillPoly(mask,triangle,255)
    mask=cv.bitwise_and(picture,mask)
    return mask 

def detection(picture):
    lines = cv.HoughLinesP(picture, rho=2, theta=np.pi/180, threshold=100, lines=np.array([]), minLineLength=40, maxLineGap=5)
    left = []
    right = []
    
    if lines is not None:  
        for line in lines:
            
            x1, y1, x2, y2 = line.reshape(4)
           
            try:
                parameters = np.polyfit((x1, x2), (y1, y2), 1)
                slope = parameters[0]
                y_int = parameters[1]
                if slope < 0:
                    left.append((slope, y_int))
                else:
                    right.append((slope, y_int))
            except Exception as e:
                print(f"Error in line fitting: {e}")
                continue  

        if left and right:  
            right_avg = np.average(right, axis=0)
            left_avg = np.average(left, axis=0)
            right_line = make_points(picture, right_avg)
            left_line = make_points(picture, left_avg)
            return np.array([left_line, right_line])
    
    return [] 
def make_points(image,average):
    slope,y_int=average
    y1=image.shape[0]
    y2=int(y1*(3/5))
    x1=int((y1-y_int)//slope)
    x2=int((y2-y_int)//slope)
    return np.array([x1,y1,x2,y2])
def display(picture,lines):
    lines_image=np.zeros((picture.shape[0], picture.shape[1],3),np.uint8)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2=line 
            cv.line(lines_image,(x1,y1),(x2,y2),(0,255,0),10)
    return lines_image

def sign_detection(frame,template):
    tem=template
    threshold=0.4
    result=cv.matchTemplate(frame,tem,cv.TM_CCOEFF_NORMED

    )
    height,width=tem.shape[:2]
    
    locations=np.where(result>=threshold)
    if locations[0].size>0:

            for pt in zip(*locations[::-1]):
                bottom_right = (pt[0] + width, pt[1] + height)
                cv.rectangle(frame, pt, bottom_right, 255, 2)
                font=cv.FONT_HERSHEY_SIMPLEX
                cv.putText(frame,text="WARNING:turn to righthand!",org=(0,50),thickness=3,fontFace=font,fontScale=2,color=(0,0,159))
    else:
        print("no sign detected!!")

    


 

image_sign=cv.imread("/home/mahan/Pictures/Screenshots/sign2.png")
try:
    video=cv.VideoCapture("/home/mahan/Videos/Screencasts/final.webm") #here is the vacancy you import your video
    while True:
        ret,frame=video.read()
        if ret:
            
            pic=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
           
            _,pic=cv.threshold(pic,200,255,cv.THRESH_BINARY)
            kernel=cv.getStructuringElement(cv.MORPH_RECT,(6,6))
            pic=cv.morphologyEx(pic,cv.MORPH_CLOSE,kernel)
            pic=cv.dilate(pic,np.ones((5,5),np.float32))
            pic=mask(pic)
            lines=detection(pic)
            detected=display(pic,lines)
            sign_detection(frame,template=image_sign)
            
            finalpic=cv.addWeighted(frame,0.8,detected,1,0)
           

            cv.imshow("video",finalpic)
            if cv.waitKey(1) & 0xFF==ord('q'):
                break

        else:
            break        

    video.release()
    cv.destroyAllWindows()                  
except Exception as e:
    print(f"faced {e}")   