from tensorflow.python.compiler.tensorrt import trt_convert as trt

import cv2
import time
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import os
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
with tf.Session(graph=graph) as sess:
    with tf.gfile.GFile(pb_file, 'rb') as f:
        frozen_graph = tf.GraphDef()
        frozen_graph.ParseFromString(f.read())
    converter = trt.TrtGraphConverter(input_graph_def=frozen_graph,max_batch_size=32,nodes_blacklist=return_elements,precision_mode='FP16',is_dynamic_op=True)  
    trt_graph = converter.convert()
