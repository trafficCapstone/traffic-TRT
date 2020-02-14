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


return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
pb_file         = "./yolov3_coco.pb"
video_path      = "video3.mp4"
# video_path      = 0
num_classes     = 80
input_size      = 608
graph           = tf.Graph()
#return_tensors  = utils.read_pb_return_tensors(graph, pb_file, return_elements)
def read_pb_files(path,graph):
    with graph.as_default():
        with tf.gfile.FastGFile(path,'rb') as f:
            graph_def=tf.GraphDef()
            graph_def.ParseFromString(f.read())
    return graph_def

def GetFP16(graph,graph_def,nodes_list):
    with graph.as_default():
        trt_graph=trt.create_inference_graph(graph_def, nodes_list,precision_mode='FP16',max_workspace_size_bytes=1<<30,max_batch_size=32)
    return trt_graph

def GetINT8(graph,graph_def,nodes_list):
    with graph.as_default():
        trt_graph=trt.create_inference_graph(graph_def, nodes_list,precision_mode='INT8',max_workspace_size_bytes=1<<30,max_batch_size=32)
    return trt_graph

with tf.Session(graph=graph) as sess:
    graph_def = read_pb_files(pb_file,graph)
    trt_graph = GetFP16(graph,graph_def,return_elements)
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
            print("total processtime:",usedtime)
            raise ValueError("No image!")
        frame_size = frame.shape[:2]
        image_data = utils.image_preporcess(np.copy(frame), [input_size, input_size])
        image_data = image_data[np.newaxis, ...]
        prev_time = time.time()

        pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
            [return_tensors[1], return_tensors[2], return_tensors[3]],
                    feed_dict={ return_tensors[0]: image_data})

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
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("result", result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break




