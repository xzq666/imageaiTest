#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 17:11:11 2018

@author: qhzc-imac-02
"""

from imageai.Detection import VideoObjectDetection
import os
import time

start = time.time()
print("开始识别")

# 当前文件目录
execution_path = os.getcwd()

detector = VideoObjectDetection()
# 设置使用的模型
detector.setModelTypeAsTinyYOLOv3()
# 加载训练好的模型数据
detector.setModelPath(os.path.join(execution_path, 'yolo-tiny.h5'))
detector.loadModel()
# 设置输入视频地址 输出地址 每秒帧数等
detector.detectObjectsFromVideo(input_file_path=os.path.join(execution_path, 'test.mp4'),
                                output_file_path=os.path.join(execution_path, 'test_detected'),
                                frames_per_second=20, log_progress=True)
"""
custom_objects = detector.CustomObjects(person=True, bicycle=True, motorcycle=True)
video_path = detector.detectCustomObjectsFromVideo(custom_objects=custom_objects, 
                                                   input_file_path=os.path.join(execution_path, "traffic.mp4"), 
                                                   output_file_path=os.path.join(execution_path, "traffic_custom_detected"), 
                                                   frames_per_second=20, log_progress=True)
"""

end = time.time()
print("识别结束")
print("识别时间: " + str(end - start))
