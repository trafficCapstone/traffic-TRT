Install Instruction:
```bashrc
OS:Ubuntu 18.04
CPU:Xeon Scaleble Gold 6139
RAM:DDR4 96G
GPU:RTX Titan
Python:3.6.9
Much more we need the CUDA 10.0,CUDNN 7.6 and Tensor RT6.0.1
First, we need the dependency library:
We can install it by PIP


 numpy>=1.18.0
 Pillow==5.3.0
 scipy==1.1.0
 tensorflow-gpu==1.14.0
 wget==3.2
 seaborn==0.9.0
```
Clone the project:
```bashrc
https://github.com/trafficCapstone/traffic-TRT.git
```
Because the tensorflow in the PIP is not support the tensorRT6 according to our experience, we provide a version that complied by ourselves:
```bashrc
https://drive.google.com/file/d/1mUv7B_X-EgjG7D3Qiqp3C1mNtimayxzS/view?usp=sharing
```
DownLoad the models From the link:
```bashrc
https://drive.google.com/file/d/1FGJHxFQf9xIjmvF39Nj3gLDReRvJtBId/view?usp=sharing
```
DownLoad the Testing video :
```bashrc
https://drive.google.com/file/d/16Gkj73lviMBXBBqzZjrFC4-kQhvk7opl/view?usp=sharing
```
Then,run the TensorRT version:
```bashrc
python3 video_demo.py 
```
The version that without the tensorRT is
```bashrc
python3  video_demo_cuda.py
```
The Frame process time with Tensor RT:
![image](https://github.com/trafficCapstone/traffic-TRT/blob/master/Wiki/40MS.PNG)
The Frame process time without Tensor RT:
![image](https://github.com/trafficCapstone/traffic-TRT/blob/master/Wiki/55ms.PNG)
The power usage on GPU with TensorRT:
![image](https://github.com/trafficCapstone/traffic-TRT/blob/master/Wiki/93w.PNG)
The power usage on GPU without TensorRT:
![image](https://github.com/trafficCapstone/traffic-TRT/blob/master/Wiki/191w.PNG)
The CPU & RAM Usage with TensorRT：
![image](https://github.com/trafficCapstone/traffic-TRT/blob/master/Wiki/93w-300cpu.PNG)
The CPU & RAM Usage without TensorRT：
![image](https://github.com/trafficCapstone/traffic-TRT/blob/master/Wiki/271-cpu.PNG)

![image](https://github.com/trafficCapstone/traffic-TRT/blob/master/Wiki/Table.PNG)

From the tables above, we can see that the tensor RT is help us reduce about 37.5% times in the YOLOv3 Predict progress and it also reduce 17.6% in the overall process of the frame.
We also found that the NMS algorithm(Mostly implement with the Numpy) and the drawbox parts that using the openCV is waste about half time is the overall frame process. After switch to the TensorRT, we can see that the CPU Load is in 300%, which is a special number that with 3 cores full load on the machine. This means that the bottlenleak was case by the part that need to using many cpu like the drawbox and the NMS algorithm. Much more, we can see that with the tensorRT we only need to use half power of the without tensorRT. The GPU load is also much lower when we using the tensorRT.

