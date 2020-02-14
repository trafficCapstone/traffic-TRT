def read_pb_files(path,graph):
    with graph.as_default():
        with gfile.FastGFile(path,'rb') as f:
            graph_def=tf.GraphDef()
            graph_def.ParseFromString(f.read())
    return graph_def

def GetFP16(graph,graph_def,nodes_list):
    with graph.as_default():
        trt_graph=trt.create_inference_graph(graph_def, nodes_list,precision_mode='FP16')
    return trt_graph
