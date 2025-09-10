import cv2
import pandas as pd
from ultralytics import YOLO
#from tracker import *
import cvzone
import numpy as np
import time


model = YOLO('yolo11n.pt') #definição do modelo pronto (s = small -> versão mais leve)
names= model.names #nomes dos objetos que o modelo consegue detectar

line_x = 278
#captura posição do mouse sobre o vídeo
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :
        point = [x, y]
        print(point) 

#captura de video da câmera
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
webcam=cv2.VideoCapture(0)

in_count = 0
out_count = 0

frame_count = 0
while True:
    ret, frame = webcam.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % 2 != 0:
        continue
    frame = cv2.resize(frame, (640, 480))

    #detect and track persons
    results = model.track(frame, persist=True, classes=[0])

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        for track_id,box,class_id in zip(ids,boxes,class_ids):
            x1,y1,x2,y2 = box
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

    cvzone.putTextRect(frame, f'IN: {in_count}', (40,60), scale = 2, thickness=2, colorT=(255, 255, 255), colorR=(0, 128, 0))
    cvzone.putTextRect(frame, f'IN: {out_count}', (40,100), scale = 2, thickness=2, colorT=(255, 255, 255), colorR=(0, 0, 255))
    time.sleep(1)
    