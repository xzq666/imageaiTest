#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 09:28:34 2018

@author: qhzc-imac-02
"""

from imageai.Detection import ObjectDetection
import os

# 返回当前目录
execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path, 'resnet50_coco_best_v2.0.1.h5'))
detector.loadModel()
detections = detector.detectCustomObjectsFromImage(input_image=os.path.join(execution_path , "image_1.jpg"), output_image_path=os.path.join(execution_path , "imagenew.jpg"))

for eachObject in detections:
    print(eachObject['name'] + ': ' + str(eachObject['percentage_probability']))
