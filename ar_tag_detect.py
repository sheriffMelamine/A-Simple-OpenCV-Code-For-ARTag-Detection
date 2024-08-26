import numpy as np
import cv2 as cv

def ARCtoNPC(img):
    img_shape=np.shape(img)[0]
    img_n=9
    dn=5
    start=(img_n-dn)*0.5
    parts=img_shape/img_n
    ret=np.zeros((dn,dn))
    for i in range(0,dn):
        for j in range(0,dn):
            a=int((start+i)*parts)
            b=int((start+i+1)*parts)
            c=int((start+j)*parts)
            d=int((start+j+1)*parts)
            m=np.sum(img[a:b,c:d].astype(np.int32))/(255*(b-a)*(d-c))
            ret[i,j]=0 if m<0.5 else 1
    return ret
    
    
def matchDICT(img1,img2):
    arr=ARCtoNPC(img1)
    arr2=ARCtoNPC(img2)
    ret=False
    for i in range(0,4):
        text=""
        arr=np.rot90(arr)
        if np.array_equal(arr,arr2):
            ret=True
            break
    return ret
    
mark=[]
img=[]
imghsv=[]
mask=[]
res=[]
thresh=[]
cnt=[]
shapes=[]
roi=[]
marks=[]
approx=[]

for i in range(0,4):
    img.append(cv.imread("AR_Tag_Task/scene"+str(i+1)+".jpg"))
    shapes.append(np.shape(img[i]))

for i in range(0,4):
    imghsv.append(cv.cvtColor(img[i],cv.COLOR_BGR2HSV)) 
    mask.append(cv.inRange(imghsv[i],np.array([0,0,150]).astype(np.uint8),np.array([179,50,255]).astype(np.uint8)))
    res.append(cv.bitwise_and(img[i],img[i],mask=mask[i]))
    res[i]=cv.cvtColor(res[i],cv.COLOR_BGR2GRAY)
    thresh.append(cv.threshold(res[i],100,255,0)[1])
    cnt.append(cv.findContours(thresh[i], cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0])

for i in range(0,4):
    for cntss in cnt[i]:
        temp1=cv.contourArea(cntss)
        if 25000<temp1:
            temp2=cv.arcLength(cntss,True)
            if 12.8*temp1<temp2**2<19.2*temp1:
                approx.append(cv.approxPolyDP(cntss,0.1*cv.arcLength(cntss,True),True))
                                                         
for i in range(0,4):
    temp=cv.getPerspectiveTransform(approx[i].astype(np.float32),np.float32([[0,0],[383,0],[383,383],[0,383]]))
    roi.append(cv.warpPerspective(thresh[i],temp,(384,384)))

for i in range(0,2):
    mark.append(cv.imread("AR_Tag_Task/marker"+str(i+1)+".jpg",0))
    marks.append((mark[i])[64:447,64:447])  

for i in range(0,4):
    text="MARKER 1" if matchDICT(marks[0],roi[i]) else "MARKER 2"
    for j in range(0,4):
        cv.line(img[i],approx[i][j,0],approx[i][j+1 if not j==3 else 0,0],(0,255,0),20)

    mom= cv.moments(approx[i])
    org=(int(mom['m10']/mom['m00'])-200,int(mom['m01']/mom['m00'])+200)
    cv.putText(img[i],text,org,cv.FONT_HERSHEY_SIMPLEX,6,(0,255,255),20,cv.LINE_AA)
for i in range(0,4):
    cv.imwrite("AR_Tag_Task/scene"+str(i+1)+"final.jpg",img[i]) 
