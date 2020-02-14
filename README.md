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
