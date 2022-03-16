import cv2 as cv

img= cv.imread("cat1.jpg")

objNames=[]
objFile= "coco.names"

with open(objFile,'rt') as f:
    objNames= f.read().rstrip('\n').split('\n')

confPath="ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightPath="frozen_inference_graph.pb"

net=cv.dnn_DetectionModel(weightPath,confPath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)

classIds, confs ,bbox =net.detect(img,confThreshold=0.5)
print(classIds,bbox)

for classId , conf, box in zip(classIds.flatten(),confs.flatten(),bbox):
    cv.rectangle(img,box,(0,255,0),2)
    cv.putText(img,objNames[classId-1].upper(),(box[0]+10,box[1]+20),cv.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0),2)


cv.imshow("Output",img)
cv.waitKey(0)