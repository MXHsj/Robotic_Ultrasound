# Overview
This project aims to develop an augmented-reality based ultrasound scanning guidance system that can be used to:
- assist sonographers unfamiliar with lung ultrasound protocal
- act as the high-level planner for fully-automated ultrasound scanning robot platform which is under development at WPI MEDFUSION lab

# Componentes

## 1. DensePose
### Installtion on Ubuntu 18.04
1. installation steps https://colab.research.google.com/github/tugstugi/dl-colab-notebooks/blob/master/notebooks/DensePose.ipynb
2. detailed notes http://linkinpark213.com/2018/11/18/densepose-minesweeping/
3. additional packges required in anaconda2:
    1. pip install chumpy
    2. pip install rospkg

## 2. RealSense D435i
### Installation on Ubuntu 18.04
1. librealsense https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md
    - Register the server's public key:
    ``` shell
    sudo apt-key adv --keyserver keys.gnupg.net --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
    ```
    - for Ubuntu 18.04
    ``` shell
    sudo add-apt-repository "deb http://realsense-hw-public.s3.amazonaws.com/Debian/apt-repo bionic main" -u
    ```
    - install libraires
    ``` shell
    sudo apt-get install librealsense2-dkms
    sudo apt-get install librealsense2-utils
    sudo apt-get install librealsense2-dev
    sudo apt-get install librealsense2-dbg
    ```

2. ROS1 wrapper https://github.com/IntelRealSense/realsense-ros
    - under catkin_ws/src
    ```shell
    git clone https://github.com/IntelRealSense/realsense-ros.git
    cd ..
    catkin_make -DCATKIN_ENABLE_TESTING=False -DCMAKE_BUILD_TYPE=Release
    catkin_make install
    source ./devel/setup.zsh
    ```

3. python wrapper 
    ``` shell
    pip3 install pyrealsense2
    ```

### running
1. read pointcloud in ROS
    ``` shell
    roslaunch realsense2_camera rs_camera.launch filters:=pointcloud
    ```
    pointcloud topic: /camera/depth/color/points

2. python wrapper tutorials: https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python/examples

# Publications
1. Bimbraw K., Ma X., Zhang Z., Zhang H. (2020) Augmented Reality-Based Lung Ultrasound Scanning Guidance. In: Hu Y. et al. (eds) Medical Ultrasound, and Preterm, Perinatal and Paediatric Image Analysis. ASMUS 2020, PIPPI 2020. Lecture Notes in Computer Science, vol 12437. Springer, Cham. https://doi.org/10.1007/978-3-030-60334-2_11
