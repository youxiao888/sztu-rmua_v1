#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from dt_apriltags import Detector
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State, PositionTarget
from mavros_msgs.srv import CommandBool, SetMode
import tf.transformations
import time
from scipy.spatial.transform import Rotation as R
import math
import traceback
import inspect

VELOCITY2D_CONTROL = 0b101111000111
class DynamicLanding:
    def __init__(self):
        rospy.init_node('dynamic_landing_node')
    
        
        self.fx = rospy.get_param('~landing_fx', 728.8460)
        self.fy = rospy.get_param('~landing_fy', 728.8460)
        self.cx = rospy.get_param('~landing_cx', 728.8460)
        self.cy = rospy.get_param('~landing_cy', 728.8460)
        self.kp = rospy.get_param('~kp', 0.4)
        self.kd = rospy.get_param('~kp', 0.1)
        self.max_vel = rospy.get_param('~max_vel', 2.0)
        
        # AprilTag 检测器
        try:
            # 尝试获取库版本信息
            lib_info = dir(Detector)
            rospy.loginfo(f"Using AprilTag detector with methods: {[m for m in lib_info if not m.startswith('_')][:5]}")
            
            # 初始化检测器
            # 修改检测器初始化参数
            self.tag_detector = Detector(families='tag36h11',
                                    nthreads=8,
                                    quad_decimate=1.5,  # 增加到1.5，减少处理分辨率
                                    quad_sigma=0.8,     # 增加到0.8，提高平滑度
                                    refine_edges=0,     # 关闭边缘精细化
                                    decode_sharpening=0.1,  # 降低锐化系数
                                    debug=0)
            rospy.loginfo("AprilTag detector initialized with standard parameters")
        except Exception as e:
            rospy.logerr(f"Error initializing AprilTag detector: {e}")
            rospy.logerr("Using simplified detector")
            self.tag_detector = Detector(families='tag36h11')
        
        # 相机内参 (根据实际相机进行调整)
        self.camera_matrix = np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ])
        print(f"Camera matrix: {self.camera_matrix}")
        self.distortion_coeffs = np.zeros((5, 1))
        
        # AprilTag 尺寸 (米) - 根据实际尺寸调整
        self.tag_size = rospy.get_param('~tag_size', 0.136)  # 从参数服务器获取，默认0.15米
        rospy.loginfo(f"Using AprilTag size: {self.tag_size} meters")
        
        # 初始化 CV bridge
        self.bridge = CvBridge()
        
        # 状态变量
        self.drone_pose = None
        self.car_pose = None
        self.car_velocity = None
        self.car_trajectory = []
        self.landing_phase = "INIT"  # INIT, SEARCH, APPROACH, FINAL_APPROACH, LANDING
        self.last_tag_detection_time = None
        self.tag_visible = False
        self.search_position = None
        self.current_yaw = 0.0
        
        # 初始悬停位置
        self.initial_hover_position = None
        
        # 模式切换和解锁控制变量
        self.offboard_mode_enabled = False
        self.armed = True
        self.mode_change_time = None
        self.last_request = rospy.Time.now()
        
        # 卡尔曼滤波器用于预测小车运动
        self.kf = cv2.KalmanFilter(6, 3)  # 状态: [x, y, z, vx, vy, vz], 测量: [x, y, z]
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], np.float32)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], np.float32)
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.03
        
        # 订阅话题
        rospy.Subscriber('/mavros/local_position/odom', Odometry, self.drone_pose_callback)
        rospy.Subscriber('/usb_cam/image_raw', Image, self.image_callback)
        rospy.Subscriber('/mavros/state', State, self.state_callback)
        
        # 发布话题
        self.pose_pub = rospy.Publisher('/mavros/setpoint_position/local', PoseStamped, queue_size=10)
        self.raw_pub = rospy.Publisher('/mavros/setpoint_raw/local', PositionTarget, queue_size=10)
        
        # 等待服务可用
        rospy.loginfo("Waiting for services...")
        rospy.wait_for_service('/mavros/cmd/arming')
        rospy.wait_for_service('/mavros/set_mode')
        
        # 服务客户端
        self.arming_client = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
        self.set_mode_client = rospy.ServiceProxy('/mavros/set_mode', SetMode)
        
        # 无人机当前状态
        self.current_state = None
        
        # 调试图像发布
        self.debug_image_pub = rospy.Publisher('/landing/debug_image', Image, queue_size=10)
        
        rospy.loginfo("Dynamic landing node initialized")
    
    def state_callback(self, msg):
        self.current_state = msg
        
        # 更新状态
        if self.current_state.mode == "OFFBOARD":
            self.offboard_mode_enabled = True
        else:
            self.offboard_mode_enabled = False
        
        self.armed = self.current_state.armed
    
    def drone_pose_callback(self, msg):
        self.drone_pose = msg.pose.pose
        
        # 记录初始悬停位置
        if self.initial_hover_position is None and self.drone_pose is not None:
            self.initial_hover_position = np.array([
                self.drone_pose.position.x,
                self.drone_pose.position.y,
                self.drone_pose.position.z
            ])
            rospy.loginfo(f"Initial hover position set to: {self.initial_hover_position}")
        
        # 从四元数提取当前yaw角
        q = [self.drone_pose.orientation.x, self.drone_pose.orientation.y,
             self.drone_pose.orientation.z, self.drone_pose.orientation.w]
        euler = tf.transformations.euler_from_quaternion(q)
        self.current_yaw = euler[2]  # 提取yaw角
        
    def get_yaw_quaternion(self, yaw):
        """将yaw角转换为四元数"""
        return tf.transformations.quaternion_from_euler(0, 0, yaw)
    
    def image_callback(self, msg):
        try:
            # 转换ROS图像为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # 创建调试图像
            debug_img = cv_image.copy()
            
            # 检测AprilTag - 确保提供所有必要参数
            try:
                # 相机参数
                camera_params = (self.camera_matrix[0, 0], self.camera_matrix[1, 1], 
                                self.camera_matrix[0, 2], self.camera_matrix[1, 2])
                
                # 检查该库是否支持 estimate_tag_pose 参数
                if hasattr(self.tag_detector, 'detect') and 'estimate_tag_pose' in inspect.signature(self.tag_detector.detect).parameters:
                    # 明确指定估计位姿
                    tags = self.tag_detector.detect(gray_image, 
                                                camera_params=camera_params,
                                                tag_size=self.tag_size,
                                                estimate_tag_pose=True)
                    rospy.logdebug("AprilTag detection with explicit pose estimation")
                else:
                    # 使用默认参数 (大多数库会自动估计位姿)
                    tags = self.tag_detector.detect(gray_image, 
                                                camera_params=camera_params,
                                                tag_size=self.tag_size)
                    rospy.logdebug("AprilTag detection with default parameters")
            except Exception as e:
                rospy.logwarn(f"Error in tag detection with full parameters: {e}")
                # 退回到最简单的调用形式
                tags = self.tag_detector.detect(gray_image)
                rospy.logwarn("Falling back to basic detection without pose estimation")
            
            if tags:
                self.tag_visible = True
                self.last_tag_detection_time = rospy.Time.now().to_sec()
                
                # 取第一个标签 (假设只有一个)
                tag = tags[0]
                
                # 检查是否成功估计了标签的位姿
                if hasattr(tag, 'pose_R') and hasattr(tag, 'pose_t') and tag.pose_R is not None and tag.pose_t is not None:
                    # 获取标签位姿 (这是相对于相机坐标系的)
                    R_tag_to_cam = tag.pose_R  # 旋转矩阵
                    t_tag_to_cam = tag.pose_t  # 平移向量
                    
                    # 下视相机到无人机坐标系的转换矩阵
                    # 对于下视相机，坐标转换为:
                    # 相机 X 轴 -> 机体 X 轴 (前)
                    # 相机 Y 轴 -> 机体 -Y 轴 (左)
                    # 相机 Z 轴 -> 机体 -Z 轴 (上)
                    R_cam_to_drone = np.array([
                        [1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]
                    ])
                    
                    # 确保R_tag_to_cam和t_tag_to_cam是正确格式的numpy数组
                    if isinstance(R_tag_to_cam, np.ndarray) and R_tag_to_cam.size > 0:
                        # 转换旋转矩阵到无人机坐标系
                        R_tag_to_drone = R_cam_to_drone @ R_tag_to_cam
                        
                        if isinstance(t_tag_to_cam, np.ndarray) and t_tag_to_cam.size > 0:
                            # 转换平移向量到无人机坐标系
                            t_tag_to_drone = R_cam_to_drone @ t_tag_to_cam
                            
                            # 转换为以无人机为参考系的小车位置
                            self.car_pose = t_tag_to_drone.flatten()
                            
                            # 更新卡尔曼滤波器
                            if len(self.car_trajectory) > 0:
                                dt = rospy.Time.now().to_sec() - self.car_trajectory[-1][0]
                                self.kf.transitionMatrix[0, 3] = dt
                                self.kf.transitionMatrix[1, 4] = dt
                                self.kf.transitionMatrix[2, 5] = dt
                            
                            measured = np.array([[self.car_pose[0]], [self.car_pose[1]], [self.car_pose[2]]], np.float32)
                            self.kf.correct(measured)
                            state = self.kf.predict()
                            
                            # 保存轨迹和速度信息
                            timestamp = rospy.Time.now().to_sec()
                            self.car_trajectory.append((timestamp, self.car_pose.copy()))
                            self.car_velocity = np.array([state[3][0], state[4][0], state[5][0]])
                        else:
                            rospy.logwarn("Tag translation vector is empty or None")
                    else:
                        rospy.logwarn("Tag rotation matrix is empty or None")
                else:
                    # 如果AprilTag库未返回位姿估计结果，手动计算
                    rospy.logwarn("No pose estimation from AprilTag detector. Computing manually.")
                    if hasattr(tag, 'corners') and tag.corners is not None and len(tag.corners) == 4:
                        # 使用标签的四个角点手动估计位置
                        # 计算标签的中心点
                        center_x = np.mean([corner[0] for corner in tag.corners])
                        center_y = np.mean([corner[1] for corner in tag.corners])
                        
                        # 估计标签距离 - 通过计算标签的像素大小并与实际大小比较
                        # 这是一个简化计算，假设标签面向相机
                        pixel_length1 = np.linalg.norm(np.array(tag.corners[0]) - np.array(tag.corners[1]))
                        pixel_length2 = np.linalg.norm(np.array(tag.corners[1]) - np.array(tag.corners[2]))
                        pixel_length3 = np.linalg.norm(np.array(tag.corners[2]) - np.array(tag.corners[3]))
                        pixel_length4 = np.linalg.norm(np.array(tag.corners[3]) - np.array(tag.corners[0]))
                        avg_pixel_length = (pixel_length1 + pixel_length2 + pixel_length3 + pixel_length4) / 4
                        
                        # 使用相似三角形估计距离
                        # 距离 = 实际尺寸 * 焦距 / 像素尺寸
                        distance = self.tag_size * self.camera_matrix[0, 0] / avg_pixel_length
                        
                        # 计算相机坐标系中的3D坐标
                        x_cam = (center_x - self.camera_matrix[0, 2]) * distance / self.camera_matrix[0, 0]
                        y_cam = (center_y - self.camera_matrix[1, 2]) * distance / self.camera_matrix[1, 1]
                        z_cam = distance
                        
                        # 转换到无人机坐标系
                        # 对于下视相机: x_drone = x_cam, y_drone = -y_cam, z_drone = -z_cam
                        self.car_pose = np.array([x_cam, -y_cam, -z_cam])
                        
                        if len(self.car_trajectory) > 0:
                            dt = rospy.Time.now().to_sec() - self.car_trajectory[-1][0]
                            self.kf.transitionMatrix[0, 3] = dt
                            self.kf.transitionMatrix[1, 4] = dt
                            self.kf.transitionMatrix[2, 5] = dt
                        
                        measured = np.array([[self.car_pose[0]], [self.car_pose[1]], [self.car_pose[2]]], np.float32)
                        self.kf.correct(measured)
                        state = self.kf.predict()
                        
                        # 保存轨迹和速度信息
                        timestamp = rospy.Time.now().to_sec()
                        self.car_trajectory.append((timestamp, self.car_pose.copy()))
                        self.car_velocity = np.array([state[3][0], state[4][0], state[5][0]])
                    else:
                        rospy.logwarn("Cannot compute tag pose: corners not available")
                
                # 绘制AprilTag检测结果
                if hasattr(tag, 'corners') and tag.corners is not None:
                    for idx in range(len(tag.corners)):
                        cv2.circle(debug_img, (int(tag.corners[idx][0]), int(tag.corners[idx][1])), 5, (0, 255, 0), -1)
                    
                    # 连接四个角点形成矩形
                    for i in range(4):
                        j = (i + 1) % 4
                        pt1 = (int(tag.corners[i][0]), int(tag.corners[i][1]))
                        pt2 = (int(tag.corners[j][0]), int(tag.corners[j][1]))
                        cv2.line(debug_img, pt1, pt2, (0, 255, 0), 2)
                
                # 绘制中心点和ID
                if hasattr(tag, 'center') and tag.center is not None:
                    center = (int(tag.center[0]), int(tag.center[1]))
                    cv2.circle(debug_img, center, 5, (0, 0, 255), -1)
                    
                    if hasattr(tag, 'tag_id'):
                        cv2.putText(debug_img, f"ID: {tag.tag_id}", (center[0] - 10, center[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # 显示距离信息（如果可用）
                if self.car_pose is not None:
                    distance = np.linalg.norm(self.car_pose)
                    height_diff = self.car_pose[2]  # Z轴差异
                    cv2.putText(debug_img, f"Distance: {distance:.2f}m, Height: {height_diff:.2f}m", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(debug_img, f"xyz:{self.car_pose}", (10, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # 显示降落阶段
                cv2.putText(debug_img, f"Phase: {self.landing_phase}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # 显示当前yaw和目标yaw
                cv2.putText(debug_img, f"Yaw: {math.degrees(self.current_yaw):.1f}", 
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # 显示模式和解锁状态
                mode_status = "OFFBOARD" if self.offboard_mode_enabled else self.current_state.mode
                arm_status = "ARMED" if self.armed else "DISARMED"
                cv2.putText(debug_img, f"Mode: {mode_status}, Status: {arm_status}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                self.tag_visible = False
                
                # 如果标签不可见但在短时间内曾经可见，使用卡尔曼滤波器预测
                if self.last_tag_detection_time is not None and \
                   rospy.Time.now().to_sec() - self.last_tag_detection_time < 2.0:
                    state = self.kf.predict()
                    self.car_pose = np.array([state[0][0], state[1][0], state[2][0]])
                    self.car_velocity = np.array([state[3][0], state[4][0], state[5][0]])
                    
                    cv2.putText(debug_img, "Tag lost: Using prediction", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    cv2.putText(debug_img, "Tag not detected", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # 显示当前yaw和目标yaw
                cv2.putText(debug_img, f"Yaw: {math.degrees(self.current_yaw):.1f}", 
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # 显示模式和解锁状态
                mode_status = "OFFBOARD" if self.offboard_mode_enabled else self.current_state.mode
                arm_status = "ARMED" if self.armed else "DISARMED"
                cv2.putText(debug_img, f"Mode: {mode_status}, Status: {arm_status}", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 发布调试图像
            self.debug_image_pub.publish(self.bridge.cv2_to_imgmsg(debug_img, "bgr8"))
            
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")
            rospy.logerr(traceback.format_exc())  # 打印完整堆栈跟踪
    
    def create_setpoint_raw_local_message(self, velocity=None, position=None, yaw=None, yaw_rate=None, ignore_position=True, ignore_velocity=True):
        """
        创建PositionTarget消息
        
        参数:
        - velocity: [vx, vy, vz] 速度命令，如果为None则忽略
        - position: [x, y, z] 位置命令，如果为None则忽略
        - yaw: 目标偏航角（弧度），如果为None则忽略
        - yaw_rate: 目标偏航角速率（弧度/秒），如果为None则忽略
        - ignore_position: 是否忽略位置控制
        - ignore_velocity: 是否忽略速度控制
        
        返回: PositionTarget消息
        """
        msg = PositionTarget()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        
        # 坐标系 - ENU
        msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        
        # 初始化类型掩码
        mask = 0
        
        # 位置控制部分
        if position is not None and not ignore_position:
            msg.position.x = position[0]
            msg.position.y = position[1]
            msg.position.z = position[2]
        else:
            # 忽略位置
            mask |= PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY | PositionTarget.IGNORE_PZ
        
        # 速度控制部分
        if velocity is not None and not ignore_velocity:
            msg.velocity.x = velocity[0]
            msg.velocity.y = velocity[1]
            msg.velocity.z = velocity[2]
        else:
            # 忽略速度
            mask |= PositionTarget.IGNORE_VX | PositionTarget.IGNORE_VY | PositionTarget.IGNORE_VZ
        
        # 忽略加速度
        mask |= PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ
        
        # 姿态控制部分
        if yaw is not None:
            msg.yaw = yaw
        else:
            mask |= PositionTarget.IGNORE_YAW
        
        if yaw_rate is not None:
            msg.yaw_rate = yaw_rate
        else:
            mask |= PositionTarget.IGNORE_YAW_RATE
        
        # 设置类型掩码
        msg.type_mask = mask
        
        return msg
    
    def control_yaw(self, yaw_error):
        """计算yaw角速度控制量"""
        # 确保yaw差值在-pi到pi范围内
        while yaw_error > math.pi:
            yaw_error -= 2 * math.pi
        while yaw_error < -math.pi:
            yaw_error += 2 * math.pi
        
        # 设置yaw角速度，使用P控制
        kp_yaw = 0.5
        yaw_rate = kp_yaw * yaw_error
        
        # 限制最大角速度
        max_yaw_rate = 0.5  # 弧度/秒
        yaw_rate = max(min(yaw_rate, max_yaw_rate), -max_yaw_rate)
        
        return yaw_rate
    
    def attempt_offboard_mode(self):
        """尝试切换到OFFBOARD模式并解锁"""
        if self.current_state is None:
            rospy.logwarn("No state information available yet. Can't switch to OFFBOARD.")
            return False
        
        current_time = rospy.Time.now()
        
        # 在尝试进入OFFBOARD模式时，持续发送当前位置
        if self.drone_pose is not None:
            # 创建位置控制消息
            pose_msg = PoseStamped()
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.header.frame_id = "map"
            
            pose_msg.pose.position.x = 0.0
            pose_msg.pose.position.y = -1.5
            pose_msg.pose.position.z = 2.0
            
            # 保持当前朝向
            pose_msg.pose.orientation = self.drone_pose.orientation
            
            # 发布位置信息
            self.pose_pub.publish(pose_msg)
        
        if self.offboard_mode_enabled:
            return True
            
        # 如果还未切换到OFFBOARD模式
        if not self.offboard_mode_enabled:
            response = self.set_mode_client(0, "OFFBOARD")
            if response.mode_sent:
                rospy.loginfo("OFFBOARD mode requested")
                rospy.set_param('/OFFBOARD_MODE', True)
            else:
                rospy.logwarn("Failed to set OFFBOARD mode")
            self.last_request = current_time
            return False
        
        # 如果未解锁且处于OFFBOARD模式
        elif not self.armed:
            response = self.arming_client(True)
            if response.success:
                rospy.loginfo("Vehicle armed")
            else:
                rospy.logwarn("Failed to arm vehicle")
            self.last_request = current_time
            return False
            
        return self.offboard_mode_enabled and self.armed
        
    def hover_at_current_position(self):
        """让无人机保持在当前位置悬停"""
        if self.drone_pose is None:
            rospy.logwarn("No drone pose information available yet. Can't hover.")
            return

        # 创建位置控制消息
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "map"
        
        # 将当前位置设为目标位置
        pose_msg.pose.position.x = self.drone_pose.position.x
        pose_msg.pose.position.y = self.drone_pose.position.y
        pose_msg.pose.position.z = self.drone_pose.position.z
        
        self.pose_pub.publish(pose_msg)
        
    def landing_control(self):
        """根据当前阶段执行降落控制"""
        if self.drone_pose is None:
            return
        
        drone_pos = np.array([
            self.drone_pose.position.x,
            self.drone_pose.position.y,
            self.drone_pose.position.z
        ])
        
        # 创建位置控制消息 (用于初始化和搜索阶段)
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "map"
        
        # 初始化阶段 - 等待无人机切换到OFFBOARD模式并解锁
        if self.landing_phase == "INIT":
            # 首先保持在当前位置悬停以稳定无人机
            self.hover_at_current_position()
            
            # 尝试切换到OFFBOARD模式并解锁
            if self.attempt_offboard_mode():
                rospy.loginfo("Successfully switched to OFFBOARD mode and armed")
                self.landing_phase = "SEARCH"
                self.search_position = np.array([
                    0.0,
                    -1.5,
                    2.0
                ])
                rospy.loginfo(f"Entering SEARCH mode at position: {self.search_position}")
        
        # 搜索阶段 - 保持位置并寻找标签
        elif self.landing_phase == "SEARCH":
            # 如果检测到AprilTag，切换到接近阶段
            if self.car_pose is not None:
                self.landing_phase = "APPROACH"
                rospy.loginfo("Tag detected! Switching to APPROACH phase")
            else:
                # 保持在搜索位置
                if self.search_position is None:
                    self.search_position = np.array([
                        0.0,
                        -1.5,
                        2.0
                    ])
                    rospy.loginfo(f"Setting search position to: {self.search_position}")
                
                # 保持在记录的位置进行搜索，同时控制yaw角
                pose_msg.pose.position.x = self.search_position[0]
                pose_msg.pose.position.y = self.search_position[1]
                pose_msg.pose.position.z = self.search_position[2]
                
                self.pose_pub.publish(pose_msg)
        
        # 当从其他模式回到搜索模式时，需要重置搜索位置
        elif self.landing_phase != "SEARCH" and self.search_position is not None:
            self.search_position = None
        
        # 接近阶段 - 向标签靠近
        elif self.landing_phase == "APPROACH":
            if self.car_pose is None:
                # 如果丢失标签，回到搜索阶段
                self.landing_phase = "SEARCH"
                rospy.loginfo("Lost tag! Switching back to SEARCH phase")
                return
            
            # 获取四元数和旋转矩阵
            q = [self.drone_pose.orientation.x, self.drone_pose.orientation.y,
                 self.drone_pose.orientation.z, self.drone_pose.orientation.w]
            r = R.from_quat(q)
            rotation_matrix = r.as_matrix()

            # 计算小车的世界坐标系位置
            car_world_pos = drone_pos + rotation_matrix @ self.car_pose
            
            # 考虑小车速度进行位置预测
            prediction_time = 0.5  # 预测0.5秒后的位置
            if self.car_velocity is not None:
                car_world_vel = rotation_matrix @ self.car_velocity
                predicted_car_pos = car_world_pos + car_world_vel * prediction_time
            else:
                predicted_car_pos = car_world_pos
            
            # 设置目标位置（在小车上方）
            target_pos = predicted_car_pos.copy()
            
            # 根据距离决定高度
            distance_to_car_xy = np.linalg.norm(target_pos[0:2] - drone_pos[0:2])
            height_above_car = max(1.0, distance_to_car_xy * 0.5)  # 水平距离越大，高度越高
            
            # 计算位置误差
            pos_error = target_pos - drone_pos

            
            # 计算速度指令
            vel_cmd = np.array([1.5 * pos_error[0], 1.5 * pos_error[1], self.kp * pos_error[2]])
            
            
            
            # 添加速度阻尼
            if self.car_velocity is not None:
                car_world_vel = rotation_matrix @ self.car_velocity
                vel_cmd[2] += self.kd * car_world_vel[2]  # 只考虑水平速度
            
            # 限制速度
            vel_norm = np.linalg.norm(vel_cmd) #欧几里得范数
            if vel_norm > self.max_vel:
                vel_cmd = self.max_vel
            print(f"{vel_cmd}")
            
            
            # 创建并发布setpoint_raw消息
            raw_msg = self.create_setpoint_raw_local_message(
                velocity=[vel_cmd[0], vel_cmd[1], vel_cmd[2]],
                ignore_position=True,
                ignore_velocity=False
            )
            self.raw_pub.publish(raw_msg)
            
            # 当接近小车并且高度适当时，切换到最终接近阶段
            # 修改条件：使用相对于小车的高度而不是绝对高度
            current_height_above_car = drone_pos[2] - target_pos[2]
            if distance_to_car_xy < 0.5 and abs(current_height_above_car)< 0.2:
                self.landing_phase = "FINAL_APPROACH"
                rospy.loginfo("Switching to FINAL_APPROACH phase")
        
        # 最终接近阶段 - 降低高度准备降落
        elif self.landing_phase == "FINAL_APPROACH":
            if self.car_pose is None and not self.tag_visible:
                # 如果在最终接近阶段丢失AprilTag且无法预测，使用最后一次已知位置
                if len(self.car_trajectory) > 0:
                    last_pos = self.car_trajectory[-1][1]
                    # 继续以最后已知位置为目标
                    q = [self.drone_pose.orientation.x, self.drone_pose.orientation.y,
                         self.drone_pose.orientation.z, self.drone_pose.orientation.w]
                    r = R.from_quat(q)
                    rotation_matrix = r.as_matrix()
                    car_world_pos = drone_pos + rotation_matrix @ last_pos
                else:
                    # 如果无法继续，回到搜索阶段
                    self.landing_phase = "SEARCH"
                    rospy.loginfo("Lost tag completely! Switching back to SEARCH phase")
                    return
            else:
                # 使用卡尔曼滤波预测的位置
                q = [self.drone_pose.orientation.x, self.drone_pose.orientation.y,
                     self.drone_pose.orientation.z, self.drone_pose.orientation.w]
                r = R.from_quat(q)
                rotation_matrix = r.as_matrix()
                car_world_pos = drone_pos + rotation_matrix @ self.car_pose
            
            # 根据小车速度进行预测
            if self.car_velocity is not None:
                car_world_vel = rotation_matrix @ self.car_velocity
                predicted_car_pos = car_world_pos + car_world_vel * 0.2  # 短期预测
            else:
                predicted_car_pos = car_world_pos
            
            # 计算降落位置（在小车上方）
            landing_pos = predicted_car_pos.copy()
            
            # 最终阶段，匀速下降
            # 计算当前相对于小车的高度
            current_height_above_car = drone_pos[2] - landing_pos[2]
            descent_rate = 1.0  # 米/秒
            
            # 使用PD控制器跟踪水平位置
            kp_xy = 1.8
            kd_xy = 0.5
            
            # 水平位置误差
            pos_error_xy = landing_pos[0:2] - drone_pos[0:2]
            
            # 计算水平速度指令
            vel_cmd_xy = kp_xy * pos_error_xy
            
            # 添加水平速度阻尼
            if self.car_velocity is not None:
                car_world_vel = rotation_matrix @ self.car_velocity
                vel_cmd_xy += kd_xy * car_world_vel[0:2]
            
            # 限制水平速度
            max_vel_xy = 2.0
            vel_norm_xy = np.linalg.norm(vel_cmd_xy)
            if vel_norm_xy > max_vel_xy:
                vel_cmd_xy = max_vel_xy
            
            # 创建并发布setpoint_raw消息
            raw_msg = self.create_setpoint_raw_local_message(
                velocity=[vel_cmd_xy[0], vel_cmd_xy[1], -descent_rate],
                ignore_position=True,
                ignore_velocity=False
            )
            self.raw_pub.publish(raw_msg)
            
            # 当非常接近时，切换到最终降落阶段（考虑标签高度偏移）
            if current_height_above_car < 0.1:  # 高度略高于标签高度偏移
                self.landing_phase = "LANDING"
                rospy.loginfo("Switching to LANDING phase")
        
        # 最终降落阶段 - 执行最终降落
        elif self.landing_phase == "LANDING":
            # 最终降落阶段，保持水平位置，垂直下降
            if self.car_pose is not None:
                q = [self.drone_pose.orientation.x, self.drone_pose.orientation.y,
                     self.drone_pose.orientation.z, self.drone_pose.orientation.w]
                r = R.from_quat(q)
                rotation_matrix = r.as_matrix()
                car_world_pos = drone_pos + rotation_matrix @ self.car_pose
                
                # 根据小车速度进行预测
                if self.car_velocity is not None:
                    car_world_vel = rotation_matrix @ self.car_velocity
                    landing_pos = car_world_pos + car_world_vel * 0.1  # 短期预测
                else:
                    landing_pos = car_world_pos
            elif len(self.car_trajectory) > 0:
                # 使用最后已知位置
                last_pos = self.car_trajectory[-1][1]
                q = [self.drone_pose.orientation.x, self.drone_pose.orientation.y,
                     self.drone_pose.orientation.z, self.drone_pose.orientation.w]
                r = R.from_quat(q)
                rotation_matrix = r.as_matrix()
                landing_pos = drone_pos + rotation_matrix @ last_pos
            else:
                # 如果完全丢失目标，继续垂直下降
                landing_pos = drone_pos.copy()
                landing_pos[2] = 0.0  # 假设地面高度为0
            
            # 设置仅追踪水平位置的PID控制
            kp_xy = 1.5
            
            # 水平位置误差
            pos_error_xy = landing_pos[0:2] - drone_pos[0:2]
            
            # 计算水平速度指令
            vel_cmd_xy = kp_xy * pos_error_xy
            
            # 限制水平速度
            max_vel_xy = 0.5
            vel_norm_xy = np.linalg.norm(vel_cmd_xy)
            if vel_norm_xy > max_vel_xy:
                vel_cmd_xy = vel_cmd_xy * (max_vel_xy / vel_norm_xy)

            # 创建并发布setpoint_raw消息 - 最终降落阶段使用较快的下降速度
            raw_msg = self.create_setpoint_raw_local_message(
                velocity=[vel_cmd_xy[0], vel_cmd_xy[1], -1.5],  # 降落速度稍微降低，更安全
                ignore_position=True,
                ignore_velocity=False
            )
            self.raw_pub.publish(raw_msg)
            
            # 当高度非常低时停止电机 (考虑标签高度偏移)
            # 使用绝对高度代替相对高度，因为此时可能已经无法看到标签
            if drone_pos[2] < (landing_pos[2] + 0.05):  # 几乎接触标签表面
                # 关闭电机
                self.arming_client(False)
                rospy.loginfo("Landing complete!")
                rospy.signal_shutdown("Landing completed successfully")
    
    def run(self):
        """主循环函数"""
        rate = rospy.Rate(60)  # 60Hz
        
        # 等待连接
        rospy.loginfo("Waiting for FCU connection...")
        while not rospy.is_shutdown() and (
                self.current_state is None or not self.current_state.connected):
            rate.sleep()
        
        rospy.loginfo("FCU connected")
        
        # 主循环
        while not rospy.is_shutdown():
            # 执行降落控制
            self.landing_control()
            
            # 保持循环频率
            rate.sleep()

if __name__ == '__main__':
    try:
        landing_controller = DynamicLanding()
        landing_controller.run()
    except rospy.ROSInterruptException:
        pass