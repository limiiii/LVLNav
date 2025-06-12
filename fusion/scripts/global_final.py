#!/home/rob/anaconda3/envs/pytorch/bin/python
# -*- coding: utf-8 -*-

import heapq
import torch
import clip
from PIL import Image
import numpy as np
import rospy
from ds_server1 import DSserver1
from ds_server2 import DSserver2
from geometry_msgs.msg import PoseStamped, Twist, Pose2D
from fusion.srv import nav_result, nav_resultResponse

from autoware_msgs.msg import PointsImage

import math
from nav_msgs.msg import Odometry  # 需要在package.xml文件中添加nav_msgs依赖
import time
import serial
import os
import socket

import sys
sys.path.insert(0, '/home/rob/lim_ws/devel/lib/python3/dist-packages')
import tf

SQRT2 = math.sqrt(2) / 2
PI = 3.14159265358979323846
kMoveAheadMatrix = np.array([[1, 0.0, -0.25], [0.0, 1.0, 0], [0.0, 0.0, 1]])
kRotateRightMatrix = np.array([[SQRT2, -SQRT2, 0.0], [SQRT2, SQRT2, 0.0], [0.0, 0.0, 1.0]])
kRotateLeftMatrix = np.array([[SQRT2, SQRT2, 0.0], [-SQRT2, SQRT2, 0.0], [0.0, 0.0, 1.0]])
kInstructionList = ["MoveAhead", "RotateLeft", "RotateRight", "LookUp", "LookDown", "Done"]
current_pose = Pose2D()
current_pose.x = 0.0
current_pose.y = 0.0
current_pose.theta = 0.0
point_cloud_msg = PointsImage()


def OdomCallBack(msg):
    global current_pose
    # linear position
    current_pose.x = msg.pose.pose.position.x
    current_pose.y = msg.pose.pose.position.y
    # quaternion to RPY conversion 四元组转欧拉角，计算西塔
    q = (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
    euler = tf.transformations.euler_from_quaternion(q)
    roll, pitch, yaw = euler
    # angular position
    current_pose.theta = yaw
    # print("current_pose: x: {} y: {} theta: {}".format(current_pose.x, current_pose.y, current_pose.theta))


class TurtleBot2Move:
    class Direction:
        kLeft = 0
        kRight = 1

    def __init__(self, movement_pub, rate):
        self.last_pose = Pose2D()
        self.current_target_pose = Pose2D()
        self.transformation_matrix = np.array([[1, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1]])
        self.rate = rate  # the larger the value, the "smoother", try a value of 1 to see "jerk" movement
        self.movement_pub = movement_pub
        #self.ser = serial.Serial('/dev/ttyACM0', 115200, timeout=5)

    def Initialize(self):
        #self.LookUp()
        print("start servers")

    def set_attribs(self, fd):
        new = termios.tcgetattr(fd)
        # new中的字段有：[iflag, oflag, cflag, lflag, ispeed, ospeed, cc]
        new[2] |= (termios.CLOCAL | termios.CREAD)
        new[2] &= ~termios.CSIZE
        new[2] |= termios.CS8
        new[2] &= ~termios.PARENB
        new[2] &= ~termios.CSTOPB
        new[2] &= ~termios.CRTSCTS
        new[0] &= ~(
                termios.IGNBRK | termios.BRKINT | termios.PARMRK | termios.ISTRIP | termios.INLCR | termios.IGNCR | termios.ICRNL | termios.IXON)
        new[3] &= ~(termios.ECHO | termios.ECHONL | termios.ICANON | termios.ISIG | termios.IEXTEN)
        new[1] &= ~termios.OPOST
        new[4] = 115200
        new[5] = 115200
        termios.tcsetattr(fd, termios.TCSANOW, new)

    def calculate_target_point(self):
        # source_point = tf.Vector3(0.0, 0.0, 1)
        # target_point = self.transformation_matrix.inverse() * source_point
        # self.current_target_pose.x = target_point.x()
        # self.current_target_pose.y = target_point.y()
        source_point = np.array([0.0, 0.0, 1.0])
        target_point = np.dot(np.linalg.inv(self.transformation_matrix), source_point)
        self.current_target_pose.x = target_point[0]
        self.current_target_pose.y = target_point[1]

        # print("target_x:", target_point.x(), "target_y:", target_point.y())

    def calculate_target_theta(self, direction):
        last_theta = self.last_pose.theta
        new_target_theta = 0.0
        if direction == self.Direction.kLeft:
            new_target_theta = last_theta + math.pi / 4
            if new_target_theta > math.pi:
                new_target_theta -= math.pi * 2
        elif direction == self.Direction.kRight:
            new_target_theta = last_theta - math.pi / 4
            if new_target_theta < -math.pi:
                new_target_theta += math.pi * 2

        self.current_target_pose.theta = new_target_theta
        # print("target_theta:", new_target_theta)

    def is_in_target_interval(self):
        global current_pose
        rospy.loginfo("start point: [(%lf %lf)] target point: [(%lf %lf)]", self.last_pose.x, self.last_pose.y,
                      self.current_target_pose.x, self.current_target_pose.y)
        distance = math.sqrt((current_pose.x - self.last_pose.x) ** 2 + (current_pose.y - self.last_pose.y) ** 2)
        if distance < 0.25:
            return True
        else:
            return False

    def TurnRight(self):
        global current_pose
        rospy.loginfo("turn right")
        self.calculate_target_theta(self.Direction.kRight)
        rospy.loginfo("target_theta: [%lf] current_theta: [%lf]", self.current_target_pose.theta, current_pose.theta)
        while not rospy.is_shutdown() and (
                current_pose.theta > self.current_target_pose.theta or current_pose.theta * self.current_target_pose.theta < 0):
            move = Twist()
            move.linear.x = 0
            move.angular.z = -0.5
            self.movement_pub.publish(move)
            # rospy.spinOnce()
            self.rate.sleep()

        actual_theta = (self.last_pose.theta - current_pose.theta) if (
                                                                                  self.last_pose.theta - current_pose.theta) > 0 else (
                    math.pi * 2 + self.last_pose.theta - current_pose.theta)
        self.last_pose = current_pose
        # rotate_right_matrix = [[math.sin(actual_theta), -math.sin(actual_theta), 0.0], [math.sin(actual_theta), math.cos(actual_theta), 0.0], [0.0, 0.0, 1]]
        # self.transformation_matrix = rotate_right_matrix * self.transformation_matrix
        rotate_right_matrix = np.array([[math.sin(actual_theta), -math.sin(actual_theta), 0.0],
                                        [math.sin(actual_theta), math.cos(actual_theta), 0.0], [0.0, 0.0, 1]])
        self.transformation_matrix = np.dot(rotate_right_matrix, self.transformation_matrix)

    def TurnLeft(self):
        global current_pose
        rospy.loginfo("turn left")
        start_turn = rospy.Time.now()
        self.transformation_matrix = np.dot(kRotateLeftMatrix, self.transformation_matrix)
        self.calculate_target_theta(self.Direction.kLeft)

        while not rospy.is_shutdown() and (
                current_pose.theta < self.current_target_pose.theta or current_pose.theta * self.current_target_pose.theta < 0):
            move = Twist()
            move.linear.x = 0
            move.angular.z = 0.5
            self.movement_pub.publish(move)
            # rospy.spinOnce()
            self.rate.sleep()

        actual_theta = (current_pose.theta - self.last_pose.theta) if (
                                                                                  current_pose.theta - self.last_pose.theta) > 0 else (
                    self.last_pose.theta - current_pose.theta - math.pi * 2)
        self.last_pose = current_pose

    def MoveAhead(self):
        global current_pose
        rospy.loginfo("move ahead")
        self.transformation_matrix = np.dot(kMoveAheadMatrix, self.transformation_matrix)
        self.calculate_target_point()

        while not rospy.is_shutdown() and self.is_in_target_interval():
            move = Twist()
            move.linear.x = 0.1  # speed value m/s
            move.angular.z = 0
            self.movement_pub.publish(move)
            # rospy.spinOnce()
            self.rate.sleep()

        self.last_pose = current_pose
        
    def GoBack(self):
        #实现后退的代码
        print("go_back")
        
    def Stop(self):
        move = Twist()
        move.linear.x = 0
        move.angular.z = 0
        self.movement_pub.publish(move)
        # rospy.spinOnce()
        self.rate.sleep()

    def LookDown(self):
        rospy.loginfo("look down")
        Write = self.ser.write(b"62")
        time.sleep(1)

    def LookUp(self):
        rospy.loginfo("look up")
        Write = self.ser.write(b"90")
        time.sleep(1)


class Solution:
    def __init__(self):
        self.inf = 5000
        ##仿真环境小屋拓扑图
        self.Gsmall_house = [[0, 7, 100, 100, 100, 100, 100, 6, 5],
                  [7, 0, 100, 7, 100, 100, 7, 9, 6],
                  [100, 100, 0, 7, 6, 11, 10, 100, 100],
                  [100, 7, 7, 0, 100, 9, 7, 100, 10],
                  [100, 100, 6, 100, 0, 7, 12, 100, 100],
                  [100, 100, 11, 9, 7, 0, 100, 100, 100],
                  [100, 7, 10, 7, 12, 100, 0, 14, 100],
                  [6, 9, 100, 100, 100, 100, 14, 0, 100],
                  [5, 6, 100, 10, 100, 100, 100, 100, 0]]
        self.G = [[0, 6, 12, 15, 18, 5, 10, 15, 3, 6],
                  [6, 0, 5, 8, 11, 11, 16, 11, 9, 5],
                  [12, 5, 0, 3, 6, 15, 10, 6, 14, 8],
                  [15, 8, 3, 0, 3, 12, 7, 3, 17, 11],
                  [18, 11, 6, 3, 0, 13, 7, 4, 20, 13],
                  [5, 11, 15, 12, 13, 0, 5, 9, 8, 12],
                  [10, 16, 10, 7, 7, 5, 0, 4, 13, 17],
                  [15, 11, 6, 3, 4, 9, 4, 0, 17, 14],
                  [3, 9, 14, 17, 20, 8, 13, 17, 0, 7],
                  [6, 5, 8, 11, 13, 12, 17, 14, 7, 0]]
                  
        self.neighbormap = {}
        self.init_neighbormap()
        # self.check_neighbor_output()

    def init_neighbormap(self):
        for i in range(len(self.G)):
            flag = False
            for j in range(len(self.G[i])):
                if flag == False and self.G[i][j] < 100 and self.G[i][j] != 0:
                    self.neighbormap[i] = [j]
                    flag = True
                elif self.G[i][j] < 100 and self.G[i][j] != 0:
                    self.neighbormap[i].append(j)
                else:
                    continue
        print("init neighbormap success!")

    def check_neighbor_output(self):
        print("输出节点及其邻居：")
        for key, value in self.neighbormap.items():
            print("key =", key, "value =", value)

    def dijkstra(self):
        n = len(self.G)  # 图的顶点数量
        result = []

        for source in range(n):
            dist = [self.inf] * n  # 存储源点到各个顶点的最短距离
            visited = [False] * n  # 标记顶点是否被访问过
            dist[source] = 0  # 源点到自身的距离为0

            # 创建一个优先队列，按照距离的增序排列，每个元素存储距离和顶点编号
            pq = [(0, source)]
            heapq.heapify(pq)

            while pq:
                u_dist, u = heapq.heappop(pq)

                if visited[u]:
                    continue

                visited[u] = True  # 将顶点标记为已访问

                # 遍历与顶点u相邻的顶点
                for v in range(n):
                    if self.G[u][v] != 0 and not visited[v]:
                        new_dist = dist[u] + self.G[u][v]  # u到v的距离

                        if new_dist < dist[v]:
                            dist[v] = new_dist  # 更新最短距离
                            heapq.heappush(pq, (dist[v], v))  # 将v加入优先队列
            result.append(dist)
        # 使用softmax对每一行进行归一化
        for i in range(len(result)):
            row = result[i]
            sumexp = sum(row)
            row[:] = [-(value / sumexp) for value in row]
        for row in result:
            for i in range(len(row)):
                row[i] *= 1.1
        return result

    def computeQ_getpath(self, cliplim, dis, startnode, orientation_list):
        goalpath = '/home/rob/lim_ws/src/fusion/scripts/pose/lab/goals.txt'
        target_pose_index = 8  ##--------------target_pose_index表示方位的初始点，以便根据初始位姿筛选导航点!!!!!!!

        n = len(cliplim[0])
        m = len(dis)
        Q = [[-self.inf] * m for _ in range(n + 1)]
        path = []
        maximun = -self.inf
        maxindex = startnode

        for i in range(m):
            Q[0][i] = dis[startnode][i]
            if maximun < dis[startnode][i]:
                maximun = dis[startnode][i]
                maxindex = i

        visited = set()
        ans_set = set()
        que = [maxindex]

        for i in range(1, n + 1):
            orientation_set = getorientation_set(goalpath, target_pose_index, orientation_list[i - 1])
            # print("orientation list:", orientation_set)

            que.append(maxindex)
            visited.add(maxindex)
            row = dis[maxindex]
            row = sorted(row)
            secondmax = row[-2]
            while que:
                nodev = que.pop(0)

                Q[i][nodev] = max(Q[i][nodev], Q[i - 1][nodev] * 0.7 + cliplim[nodev][i - 1])
                neighbor = self.neighbormap[nodev]

                for neighbor_k in neighbor:
                    Q[i][nodev] = max(Q[i][nodev], Q[i][neighbor_k] * 0.8 + dis[nodev][neighbor_k])
                    if neighbor_k not in visited:
                        que.append(neighbor_k)
                        visited.add(neighbor_k)
                if (nodev == maxindex):
                    Q[i][nodev] += secondmax / 3

            maximun = -self.inf
            curindex = 0
            for k in range(m):
                if k == 0:
                    if k != maxindex and k in orientation_set and maximun < Q[i][k]:
                        maximun = Q[i][k]
                        curindex = k
                # 判断是否在方位指示的解集之中，概率更大且跟上一点不同则更新最大值
                elif k not in ans_set and k in orientation_set and maximun < Q[i][k]:
                    maximun = Q[i][k]
                    curindex = k
            maxindex = curindex
            ans_set.add(maxindex)

            if len(visited) == m:
                print("Q数组第", i, "行计算成功！本次循环使Q取得最大值的节点为-----", maxindex)
                target_pose_index = maxindex
                path.append(maxindex)

            visited.clear()

        print("检查计算的Q值：")
        for row in Q:
            print(*("{:.5f}".format(x) for x in row))

        return path


# 记录前端选取的节点的位姿及其对应图片
def get_image_pose_dict(path):
    image_pose_dict = {}
    with open(path, 'r') as f:
        count = 0
        for line in f:
            room_list = []
            data = line.split()
            room_list.append(float(data[0]))
            room_list.append(float(data[1]))
            room_list.append(float(data[2]))
            room_list.append(float(data[3]))
            image_path = '/home/rob/lim_ws/src/fusion/scripts/images/lab/target_image' + str(count) + '.png'
            image_pose_dict[image_path] = room_list
            count += 1
    print("create dict success!")
    return image_pose_dict


def get_probs_array(item_list, image_pose_dict):
    probs_arrays = []
    device = "cpu"

    # allitem_list = ["refrigerator","cabinet", "bed", "chair", "couch", "table", "door", "window",
    #  "bookshelf", "picture", "blinds", "shelves", "curtain", "dresser", "pillow", "mirror", "clothes",
    #  "books", "television", "paper", "towel", "shower", "box", "night", "lamp", "bag", "bar", "hallway", "keyboard", "fan", "sofa"]

    for index, image_dir in enumerate(image_pose_dict):
        model, preprocess = clip.load("ViT-B/32", device=device)
        image = preprocess(Image.open(image_dir)).unsqueeze(0).to(device)
        text = clip.tokenize([item for item in item_list]).to(device)

        # text = clip.tokenize([item for item in allitem_list]).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            # probs = probs[:, [allitem_list.index(item) for item in item_list]]
            probs = probs.flatten().tolist()
            print(f"第{index + 1}次循环, Label probs ={probs}")
            probs_arrays.append(probs)

    probs_array = np.vstack(probs_arrays)
    return probs_array


def pub_pose(target_pose_list, last_target):
    global point_cloud_msg
    # 接受目标列表、局部路径目标、导航动作代理；
    target_dict = {'AlarmClock': 'clock', 'Book': 'book', 'Bowl': 'bowl', 'CellPhone': 'cell phone', 'Chair': 'chair', 'Fridge': 'refrigerator', 'Laptop': 'laptop', 'Microwave': 'microwave', 'Pot': 'pot', 'Sink': 'sink', 'Television': 'tv', 'Toaster': 'toaster', 'Fan': 'fan', 'Bookshelf': 'bookshelf', 'Bottle': 'bottle', 'Suitcase': 'suitcase', 'Person': 'person'}

    yolo_target = target_dict[last_target]
    r = rospy.Rate(1)
    R = rospy.Rate(0.15)
    for index, target_pose in enumerate(target_pose_list):
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = target_pose[0]
        pose.pose.position.y = target_pose[1]
        pose.pose.position.z = 0.0
        pose.pose.orientation.z = target_pose[2]
        pose.pose.orientation.w = target_pose[3]
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pub.publish(pose)
        res_result = nav_resultResponse()
        # res_photo = get_photoResponse()
        is_first = True
        while True:
            # 到达终点时候推出循环，否则一直等待
            if is_first == True:
                # 为避免接受上次导航结束时未及时更新的信息，首次导航时添加个间歇时间5s
                print("正在前往第{}个目标.".format(index + 1))
                R.sleep()
                is_first = False
            r.sleep()
            res_result = target_client.call()
            if res_result.isget == True:
                # 到达目标点，保存关联模型训练数据、保存图片
                print("get destination :", index)
                R.sleep()
                break
    print("Navigation completed")


def getorientation_set(path, target_pose_index, orientation):
    poselist = []
    with open(path, 'r') as f:
        for line in f:
            room_list = []
            data = line.split()
            room_list.append(float(data[0]))
            room_list.append(float(data[1]))
            room_list.append(float(data[2]))
            room_list.append(float(data[3]))
            poselist.append(room_list)
    targetpose = poselist[target_pose_index]
    zwpose_target = [0, 0, targetpose[2], targetpose[3]]
    xypose_target = [targetpose[0], targetpose[1]]
    rpy = tf.transformations.euler_from_quaternion(zwpose_target)
    theta = rpy[2]  # 获得偏航角y
    # print("偏航角为：{}".format(theta))
    result = []
    for index, item in enumerate(poselist):
        item = item[:2]
        if orientation == '0':
            result.append(index)
            continue
        if item == xypose_target:
            continue
        xypose = np.array([item[0], item[1]])
        TransformM = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        xypose_targetnp = np.array(xypose_target)
        # 通过公式计算获得B在A下的坐标pose
        pose = np.dot(TransformM, (xypose - xypose_targetnp))
        # print("转换后的pose为:{}".format(pose))
        if orientation == '1':
            if pose[0] >= 0:  # 正前方
                result.append(index)
        elif orientation == '2':  # 正后方
            if pose[0] <= 0:
                result.append(index)
        elif orientation == '3':  # 正左方
            if pose[1] >= 0:
                result.append(index)
        elif orientation == '4':  # 正右方
            if pose[1] <= 0:
                result.append(index)
    return result


def point_cloud_callback(msg):
    global point_cloud_msg
    point_cloud_msg = msg


if __name__ == '__main__':
    rospy.init_node('test_main')
    rospy.sleep(1)
    gpt = DSserver1()
    gpt2 = DSserver2()

    pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=5)
    target_client = rospy.ServiceProxy("/gettarget", nav_result)
    target_client.wait_for_service()

    path = '/home/rob/lim_ws/src/fusion/scripts/pose/lab/goals.txt'
    # 初始化所有节点
    image_pose_dict = get_image_pose_dict(path)
    language_instruction = input("input your instruction:")
    target_list = gpt.chat_with_gpt(language_instruction)
    orientation_list = gpt2.chat_with_gpt(language_instruction)
    print("Deepseek:", target_list)
    print("orientation_list:", orientation_list)
    target_pose_list = []
    # （语义目标， 图片）-->clip网络输出；(网络输出概率，最短路径dis)论文方法加权概率与距离-->
    clipresult = get_probs_array(target_list, image_pose_dict)
    s = Solution()
    dis = s.dijkstra()

    # 检查dis是否正确初始化，Q计算是否正确
    print("输出距离代价矩阵：")
    for row in dis:
        print(*("{:.4f}".format(x) for x in row))
    # 全局路径规划算法中的方位优化、导航点规划
    path = s.computeQ_getpath(clipresult, dis, 8, orientation_list)

    # 获取目标物一列中概率最大的行索引，行索引的值对应着“图像/导航点”
    for index, target in enumerate(path):
        image_path = '/home/rob/lim_ws/src/fusion/scripts/images/lab/target_image' + str(target) + '.png'
        target_pose_list.append(image_pose_dict[image_path])


    # 执行导航，结束后执行局部路径规划
    pub_pose(target_pose_list, target_list[-1])
