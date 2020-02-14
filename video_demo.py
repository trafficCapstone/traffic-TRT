#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2018 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : video_demo.py
#   Author      : YunYang1994
#   SecondTimeDev:Csedge
#   Created date: 2020-1-28
#   Description :This version add the tensorRT support so it will runfaster than before
#
#================================================================

import cv2
import time
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import os
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
pb_file         = "./yolov3_coco.pb"
video_path      = "video3.mp4"
# video_path      = 0
num_classes     = 80
input_size      = 960
graph           = tf.Graph()
#return_tensors  = utils.read_pb_return_tensors(graph, pb_file, return_elements)

count=0
with tf.Session(graph=graph) as sess:
    with tf.gfile.GFile(pb_file, 'rb') as f:
        frozen_graph = tf.GraphDef()
        frozen_graph.ParseFromString(f.read())
    converter = trt.TrtGraphConverter(input_graph_def=frozen_graph,max_batch_size=32,nodes_blacklist=return_elements,precision_mode='FP16',is_dynamic_op=True)  
    trt_graph = converter.convert()
    return_tensors = tf.import_graph_def(trt_graph,return_elements=return_elements)
    vid = cv2.VideoCapture(video_path)
    starttime=time.time()
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            usedtime=time.time()-starttime
            fps=count/usedtime
            print("total processtime:",usedtime)
            print("AvgFPS=",fps)
            raise ValueError("No image!")
        frame_size = frame.shape[:2]
        image_data = utils.image_preporcess(np.copy(frame), [input_size, input_size])
        image_data = image_data[np.newaxis, ...]
        prev_time = time.time()

        pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
            [return_tensors[1], return_tensors[2], return_tensors[3]],
                    feed_dict={ return_tensors[0]: image_data})
        curr_time = time.time()
        exec_time = curr_time - prev_time
        info = "time: %.2f ms" %(1000*exec_time)
        print("FramTimeINTensorFlow=",info)
        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

        bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, 0.3)
        bboxes = utils.nms(bboxes, 0.45, method='nms')
        image = utils.draw_bbox(frame, bboxes)

        curr_time = time.time()
        exec_time = curr_time - prev_time
        result = np.asarray(image)
        info = "time: %.2f ms" %(1000*exec_time)
        print("FramTime=",info)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("result", result)
        count=count+1
        usedtime=time.time()-starttime
        fps=count/usedtime
      #  print("AvgFPS=",fps)
        if cv2.waitKey(1) & 0xFF == ord('q'): break




