# 语言视觉激光多模态融合的机器人导航方法


## 运行时务必请参考本文档

## 环境简介
机器人运行环境：Ubuntu18.04、ROS1 melodic
主机ip地址：rob@10.3.51.200   密码：123456

目前可行的调试主机环境：Ubuntu20.04、ROS1 Noetic（用于连接ROS1网络给初始位姿）



## 运行方法
###打开usb相机、激光雷达
```
cd lim_ws/
source devel/setup.bash
roslaunch lego_loam usb_cam1.launch

cd lslidar_ros_v4
source ./devel/setup.bash
roslaunch lslidar_driver lslidar_c16.launch
```

###打开点云过滤节点
```
cd lim_ws/
source devel/setup.bash
roslaunch lego_loam run.launch
```

###打开yolov5_ros推理节点、聚类节点
```
cd lim_ws/
source devel/setup.bash
roslaunch yolov5_ros yolov5.launch
```

###打开autoware融合工具可视化
```
cd autoware.ai/
source install/setup.bash
roslaunch runtime_manager runtime_manager.launch
```
###进入sensing后选择Calibration Publisher
###Camera ID=/usb_cam target_frame=velodyne
###点击Ref选择标定获得的xml文件!!!关键
###点击RViz后Add New Panel---->ImageViewerPlugin




---------------------------------------------------------------------------
###连接小车主机ip：rob@10.3.51.200 密码123456


###启动导航功能，rviz确定初始位姿
```
cd lingao_3D_16lines/lingaonav_ws
source ./devel/setup.bash
roslaunch car_2dnav lingao_2dnav.launch path:=/home/rob/mapping/realworld/final1_bt.bt
```
###打开rviz给定初始位姿
```
rviz -d lslidar.rviz
```
###打开服务节点
```
rosrun fusion servers.py
```
###打开推理主程序
```
rosrun fusion final_main.py
```

在主程序final _main.py中键入指令示例：Get to the cabinet in front of you, then turn right pass the suitcase and stop by the chair.
执行导航
正确方位[1, 4, 0] 正确结果导航点[0, 9, 1] 
