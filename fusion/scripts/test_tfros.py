#!/home/rob/anaconda3/envs/pytorch/bin/python
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import Twist, Pose, PoseStamped
#import math
#import sys
#sys.path.insert(0, '/home/rob/lim_ws/devel/lib/python3/dist-packages')
#import tf

if __name__ == '__main__':
    rospy.init_node('point_transformer')
    pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=5)
    #rospy.sleep(2.0)
    mygoal = PoseStamped()
    mygoal.header.frame_id = 'map'
    mygoal.header.stamp = rospy.Time.now()
    
    with open('/home/rob/lim_ws/src/fusion/scripts/pose/lab/goals.txt', 'r') as f:
        count = 0
        for line in f:
            data = line.split()
            count = count + 1
            rospy.sleep(2.0)
            mygoal.pose.position.x = float(data[0])
            mygoal.pose.position.y = float(data[1])
            mygoal.pose.orientation.z = float(data[2])
            mygoal.pose.orientation.w = float(data[3])
            pub.publish(mygoal)
            print("pub pose success:", count)
            rospy.sleep(20.0)

    print("pub pose success!")

    
    rospy.spin()
    # listener = tf.TransformListener()
    # listener.waitForTransform('map', 'base_link', rospy.Time(), rospy.Duration(4.0))
    # 
    # base_link_point = PointStamped()
    # base_link_point.header.frame_id = "base_link"
    # base_link_point.header.stamp = rospy.Time.now()
    # base_link_point.point.x = 3.0
    # base_link_point.point.y = 0.5
    # # 必要的等待tf树回传坐标变换时间！
    # rospy.sleep(0.5)
    # # 将 base_link 下的坐标转换为 map 下的坐标
    # map_point = listener.transformPoint('map', base_link_point)
    # 
    # rospy.loginfo("Base_link coordinates: {}".format(base_link_point))
    # print(map_point.point.x)
    # print(map_point.point.y)
    # print(map_point.point.z)
    #print("---------test for quaternion_from_euler--------")
    #x = 100
    #y = 320
    #angle_radians = math.atan(x / y)
    #theta = math.degrees(angle_radians)
    #q1 = tf.transformations.quaternion_from_euler(0, 0, theta)
    #print(q1)
    #print("---------test for euler_from_quaternion--------")
    #q2 = (0.0, 0.0, -0.68896, 0.72480)
    #euler = tf.transformations.euler_from_quaternion(q2)
    #roll, pitch, yaw = euler
    #print("yaw:", yaw)
    
    #测试pose/goals.txt中的各个位姿都对应实验室哪个点
#1对应
