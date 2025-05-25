import cv2
import pandas as pd
from ultralytics import YOLO
#from tracker import *
import cvzone
import numpy as np


model = YOLO('yolo11s.pt') #definição do modelo pronto (s = small -> versão mais leve)

#captura posição do mouse sobre o vídeo
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :
        point = [x, y]
        print(point) 

#captura de video da câmera
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap=cv2.VideoCapture()