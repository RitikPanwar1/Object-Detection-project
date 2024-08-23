import cv2
import matplotlib.pyplot as plt
import time

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

model = cv2.dnn_DetectionModel(frozen_model,config_file)

classLabels = []
file_name = 'labels.txt'
with open(file_name,'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

print(classLabels)
print(len(classLabels))

model.setInputSize(320,320)
model.setInputScale(1.0/127/5)
model.setInputMean((127.5,127,5,127.5))
model.setInputSwapRB(True)

#photo
img = cv2.imread('road.jpg')
plt.imshow(img)

ClassIndex, confidece, bbox = model.detect(img,confThreshold =0.55)

print(ClassIndex)

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidece.flatten(),bbox):
    cv2.rectangle(img,boxes,(255,0,0),2)
    cv2.putText(img,classLabels[ClassInd-1], (boxes[0]+10,boxes[1]+40),font, fontScale = font_scale,color=(0,255,0), thickness=3)

plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows() 


#video
cap = cv2.VideoCapture('road.mp4')
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError('Error while processing with the video')

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

while True:
    ret, frame = cap.read()
    if not ret:
        break

    ClassIndex, confidence, bbox = model.detect(frame, confThreshold=.35)

    if len(ClassIndex) != 0:
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if ClassInd <= 80:
                cv2.rectangle(frame, (boxes[0], boxes[1]), (boxes[0]+boxes[2], boxes[1]+boxes[3]), (255, 0, 0), 2)
                cv2.putText(frame, classLabels[ClassInd-1], (boxes[0]+10, boxes[1]+40), font, fontScale=font_scale, color=(0, 255, 0), thickness=3)

    cv2.imshow('object detection for autonomous vehicles', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    time.sleep(.1)
cap.release()
cv2.destroyAllWindows()