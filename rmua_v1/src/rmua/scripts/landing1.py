#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
from collections import deque
import time
import math

# ROS消息类型
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from mavros_msgs.msg import State, PositionTarget
from mavros_msgs.srv import CommandBool, SetMode, CommandLong
from cv_bridge import CvBridge

# AprilTags检测
from dt_apriltags import Detector

class DynamicLandingController:
    def __init__(self):
        rospy.init_node('dynamic_landing_controller')
        
        # 初始化参数
        self.load_parameters()
        
        # 状态变量
        self.current_state = State()
        self.current_position = None
        self.position_buffer = deque(maxlen=10)  # 存储位置信息
        self.bridge = CvBridge()
        
        # AprilTag检测器
        self.detector = Detector(families='tag36h11')
        
        # 任务状态
        self.mission_state = "INIT"  # INIT, TO_SEARCH, OBSERVE, WAIT_POSITION, WAIT, LAND
        self.apriltag_positions = []  # 记录观察到的AprilTag位置
        self.observe_start_time = None
        self.leftmost_pos = None
        self.rightmost_pos = None
        self.target_wait_position = None
        self.last_apriltag_velocity = 0.0
        self.wait_at_position = None  # "LEFT" 或 "RIGHT"
        self.wait_apriltag_positions = []  # WAIT状态下的AprilTag位置记录
        self.wait_start_time = None
        
        # 发布者
        self.local_pos_pub = rospy.Publisher('/mavros/setpoint_position/local', PoseStamped, queue_size=10)
        self.setpoint_raw_pub = rospy.Publisher('/mavros/setpoint_raw/local', PositionTarget, queue_size=10)
        self.image_pub = rospy.Publisher('/processed_image', Image, queue_size=1)
        
        # 订阅者
        self.state_sub = rospy.Subscriber('/mavros/state', State, self.state_callback)
        self.position_sub = rospy.Subscriber('/mavros/local_position/odom', Odometry, self.position_callback)
        self.image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.image_callback)
        
        # 服务客户端
        self.arming_client = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
        self.set_mode_client = rospy.ServiceProxy('/mavros/set_mode', SetMode)
        self.command_client = rospy.ServiceProxy('/mavros/cmd/command',CommandLong)
        self.cmd = CommandLong()
        self.cmd.command = 400
        self.cmd.param1 = 0
        self.cmd.param2 = 21196
        
        rospy.loginfo("Dynamic Landing Controller initialized")
        
    def load_parameters(self):
        """从参数服务器加载参数"""
        # 初始位置
        self.init_pos_x = rospy.get_param('~init_pos_x', 1.5)
        self.init_pos_y = rospy.get_param('~init_pos_y', 6.9)
        self.init_pos_z = rospy.get_param('~init_pos_z', 3.5)
        
        # 相机内参
        self.camera_matrix = np.array([
            [rospy.get_param('~fx', 722.971756), 0, rospy.get_param('~cx', 359.185405)],
            [0, rospy.get_param('~fy', 719.232044), rospy.get_param('~cy', 227.504912)],
            [0, 0, 1]
        ])
        
        
        # 相机到机体的旋转矩阵（相机向下）
        self.camera_to_body_rotation = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])
        
        # AprilTag尺寸
        self.tag_size = rospy.get_param('~apriltag_size', 0.20)
        
        # 任务参数
        self.observe_height = 2.5
        self.wait_height = 1.5
        self.landing_speed = 1.3
        self.platform_size = 0.4
        self.observe_duration = 12.0
        
    def state_callback(self, msg):
        self.current_state = msg
        
    def position_callback(self, msg):
        """位置回调，维护位置缓冲区"""
        position_data = {
            'timestamp': msg.header.stamp,
            'position': msg.pose.pose.position,
            'orientation': msg.pose.pose.orientation
        }
        self.position_buffer.append(position_data)
        self.current_position = position_data
        
    def get_position_at_timestamp(self, target_timestamp):
        """根据时间戳获取最接近的位置信息"""
        if not self.position_buffer:
            return None
            
        min_diff = float('inf')
        closest_position = None
        
        for pos_data in self.position_buffer:
            time_diff = abs((pos_data['timestamp'] - target_timestamp).to_sec())
            if time_diff < min_diff:
                min_diff = time_diff
                closest_position = pos_data
                
        return closest_position
        
    def image_callback(self, msg):
        """图像处理回调"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # 检测AprilTag
            tags = self.detector.detect(gray, estimate_tag_pose=True, 
                                      camera_params=[self.camera_matrix[0,0], 
                                                   self.camera_matrix[1,1],
                                                   self.camera_matrix[0,2], 
                                                   self.camera_matrix[1,2]], 
                                      tag_size=self.tag_size)
            
            # 获取对应时间戳的位置信息
            position_data = self.get_position_at_timestamp(msg.header.stamp)
            
            if tags and position_data:
                self.process_apriltag(tags[0], position_data, cv_image)
            
            # 可视化
            self.visualize_image(cv_image, tags)
            
        except Exception as e:
            rospy.logerr(f"Image processing error: {e}")
            
    def process_apriltag(self, tag, position_data, image):
        """处理检测到的AprilTag"""
        # 计算AprilTag在全局坐标系中的位置
        tag_global_pos = self.calculate_tag_global_position(tag, position_data)
        
        if self.mission_state == "TO_SEARCH":
            rospy.loginfo("AprilTag detected! Starting observation phase")
            self.mission_state = "OBSERVE"
            self.observe_start_time = rospy.Time.now()
            self.apriltag_positions = []
            
        elif self.mission_state == "OBSERVE":
            self.apriltag_positions.append({
                'time': rospy.Time.now(),
                'position': tag_global_pos
            })
            
            # 检查观察时间是否足够
            if (rospy.Time.now() - self.observe_start_time).to_sec() >= self.observe_duration:
                self.analyze_apriltag_motion()
                
        elif self.mission_state == "WAIT":
            # 记录WAIT状态下的AprilTag位置
            self.wait_apriltag_positions.append({
                'time': rospy.Time.now(),
                'position': tag_global_pos
            })
            
            # 检查是否刚进入WAIT状态（第一帧检测到AprilTag）
            if len(self.wait_apriltag_positions) == 1:
                rospy.loginfo("First AprilTag detection in WAIT state, checking motion direction...")
                
            # 当有足够的数据点时检查运动方向
            elif len(self.wait_apriltag_positions) >= 3:
                current_velocity = self.calculate_current_velocity()
                
                # 检查运动方向是否正确
                direction_correct = self.check_motion_direction(current_velocity)
                
                if direction_correct:
                    # 方向正确，检查是否应该开始降落
                    if self.should_start_landing(tag_global_pos):
                        rospy.loginfo("Starting landing sequence")
                        self.mission_state = "LAND"
                    else:
                        rospy.loginfo("Landing conditions not met")
                else:
                    rospy.loginfo("Waiting for correct motion direction...")
                    
    def check_motion_direction(self, velocity):
        """检查AprilTag运动方向是否正确"""
        
        if self.wait_at_position == "RIGHT":
            # 在右边等待，期望AprilTag向右运动（负速度）
            return velocity < 0.0
        elif self.wait_at_position == "LEFT":
            # 在左边等待，期望AprilTag向左运动（正速度）
            return velocity > 0.02
        
        return False
        
    def calculate_current_velocity(self):
        """计算当前AprilTag速度"""
        if len(self.wait_apriltag_positions) < 2:
            return 0.0
            
        recent_positions = self.wait_apriltag_positions[-3:]  # 最近3个位置
        if len(recent_positions) >= 2:
            dt = (recent_positions[-1]['time'] - recent_positions[0]['time']).to_sec()
            dy = recent_positions[-1]['position'][1] - recent_positions[0]['position'][1]
            return dy / dt if dt > 0 else 0.0
        return 0.0
        
    def calculate_tag_global_position(self, tag, position_data):
        """计算AprilTag在全局坐标系中的位置"""
        # 从AprilTag姿态获取位置（相机坐标系）
        tag_pos_camera = tag.pose_t.flatten()
        
        # 转换到机体坐标系
        tag_pos_body = self.camera_to_body_rotation @ tag_pos_camera
        
        # 获取无人机位置和姿态
        drone_pos = position_data['position']
        drone_quat = position_data['orientation']
        
        # 转换四元数到旋转矩阵
        drone_rotation = self.quaternion_to_rotation_matrix(drone_quat)
        
        # 转换到全局坐标系
        tag_pos_global = np.array([drone_pos.x, drone_pos.y, drone_pos.z]) + drone_rotation @ tag_pos_body
        
        return tag_pos_global
        
    def quaternion_to_rotation_matrix(self, quat):
        """四元数转旋转矩阵"""
        w, x, y, z = quat.w, quat.x, quat.y, quat.z
        return np.array([
            [1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)]
        ])
        
    def analyze_apriltag_motion(self):
        """分析AprilTag运动模式"""
        if len(self.apriltag_positions) < 2:
            rospy.logwarn("Not enough AprilTag observations")
            return
            
        # 找到最左和最右位置
        y_positions = [pos['position'][1] for pos in self.apriltag_positions]
        self.leftmost_pos = max(y_positions)
        self.rightmost_pos = min(y_positions)
        
        # 计算当前速度
        recent_positions = self.apriltag_positions[-5:]  # 最近5个位置
        if len(recent_positions) >= 2:
            dt = (recent_positions[-1]['time'] - recent_positions[0]['time']).to_sec()
            dy = recent_positions[-1]['position'][1] - recent_positions[0]['position'][1]
            self.last_apriltag_velocity = dy / dt if dt > 0 else 0.0
            
        # 决定等待位置 - 在AprilTag即将到达的极值点等待
        if self.last_apriltag_velocity > 0:  # 向左运动，在右边等待
            self.target_wait_position = [self.apriltag_positions[-1]['position'][0], 
                                       self.rightmost_pos, self.wait_height]
            self.wait_at_position = "RIGHT"
        else:  # 向右运动，在左边等待
            self.target_wait_position = [self.apriltag_positions[-1]['position'][0], 
                                       self.leftmost_pos, self.wait_height]
            self.wait_at_position = "LEFT"
            
        rospy.loginfo(f"Motion analysis complete. Leftmost: {self.leftmost_pos:.2f}, "
                     f"Rightmost: {self.rightmost_pos:.2f}, "
                     f"Velocity: {self.last_apriltag_velocity:.2f}, "
                     f"Wait at: {self.wait_at_position}, "
                     f"Wait position: {self.target_wait_position}")
        
        self.mission_state = "WAIT_POSITION"
        self.wait_apriltag_positions = []
        
    def should_start_landing(self, tag_global_pos):
        """判断是否应该开始降落"""
        if not self.current_position:
            return False
            
        # 计算无人机当前位置
        drone_pos = np.array([
            self.current_position['position'].x,
            self.current_position['position'].y,
            self.current_position['position'].z
        ])
        
        # 预测降落时间
        landing_time = (drone_pos[2] - tag_global_pos[2]) / self.landing_speed
        
        # 使用当前实时速度预测AprilTag未来位置
        current_velocity = self.calculate_current_velocity()
        predicted_tag_pos = tag_global_pos + np.array([0, current_velocity * landing_time, 0])
        
        # 检查是否在平台范围内
        horizontal_distance = np.linalg.norm(drone_pos[:2] - predicted_tag_pos[:2])
        print(horizontal_distance)
        
        return horizontal_distance <= self.platform_size / 2
        
    def visualize_image(self, image, tags):
        """可视化图像"""
        vis_image = image.copy()
        
        # 绘制AprilTag检测结果
        for tag in tags:
            corners = tag.corners.astype(int)
            cv2.polylines(vis_image, [corners], True, (0, 255, 0), 2)
            cv2.putText(vis_image, f"ID: {tag.tag_id}", 
                       tuple(corners[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        # 显示运动范围和状态信息
        if self.leftmost_pos is not None and self.rightmost_pos is not None:
            info_text = f"Range: {self.leftmost_pos:.2f} to {self.rightmost_pos:.2f}"
            cv2.putText(vis_image, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        cv2.putText(vis_image, f"State: {self.mission_state}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if hasattr(self, 'wait_at_position') and self.wait_at_position:
            cv2.putText(vis_image, f"Wait at: {self.wait_at_position}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        # 显示当前速度信息
        if self.mission_state == "WAIT" and len(self.wait_apriltag_positions) >= 3:
            current_vel = self.calculate_current_velocity()
            direction_ok = self.check_motion_direction(current_vel)
            vel_text = f"Vel: {current_vel:.2f}, Dir OK: {direction_ok}"
            cv2.putText(vis_image, vel_text, (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if direction_ok else (0, 0, 255), 2)
        
        # 发布处理后的图像
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(vis_image, "bgr8"))
        except Exception as e:
            rospy.logerr(f"Failed to publish image: {e}")
            
    def send_velocity_command(self, vx, vy, vz):
        """发送速度控制指令"""
        target = PositionTarget()
        target.header.stamp = rospy.Time.now()
        target.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        
        # 设置控制模式为速度控制
        target.type_mask = (PositionTarget.IGNORE_PX | 
                           PositionTarget.IGNORE_PY | 
                           PositionTarget.IGNORE_PZ |
                           PositionTarget.IGNORE_AFX | 
                           PositionTarget.IGNORE_AFY | 
                           PositionTarget.IGNORE_AFZ |
                           PositionTarget.FORCE | 
                           PositionTarget.IGNORE_YAW | 
                           PositionTarget.IGNORE_YAW_RATE)
        
        target.velocity.x = vx
        target.velocity.y = vy
        target.velocity.z = vz
        
        self.setpoint_raw_pub.publish(target)
        
    def send_position_command(self, x, y, z):
        """发送位置控制指令"""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.pose.position.x = x
        pose_msg.pose.position.y = y
        pose_msg.pose.position.z = z
        self.local_pos_pub.publish(pose_msg)
        
    def wait_for_offboard_mode(self):
        """等待进入OFFBOARD模式"""
        rate = rospy.Rate(30)
        
        # 发布初始位置作为前置条件
        rospy.loginfo("Publishing initial position for OFFBOARD mode...")
        for i in range(100):
            if rospy.is_shutdown():
                return False
            self.send_position_command(self.init_pos_x, self.init_pos_y, self.init_pos_z)
            rate.sleep()
            
        # 设置OFFBOARD模式
        if self.current_state.mode != "OFFBOARD":
            rospy.loginfo("Setting OFFBOARD mode...")
            if self.set_mode_client(custom_mode="OFFBOARD").mode_sent:
                rospy.loginfo("OFFBOARD mode set successfully")
            else:
                rospy.logerr("Failed to set OFFBOARD mode")
                return False
                
        return True
        
    def control_loop(self):
        """主控制循环"""
        rate = rospy.Rate(20)
        
        while not rospy.is_shutdown():
            if self.mission_state == "INIT":
                if self.wait_for_offboard_mode():
                    self.mission_state = "TO_SEARCH"
                    rospy.loginfo("Flying to search position...")
                    
            elif self.mission_state == "TO_SEARCH":
                if self.current_position:
                    # 计算到初始搜索点的距离
                    current_pos = np.array([
                        self.current_position['position'].x,
                        self.current_position['position'].y,
                        self.current_position['position'].z
                    ])
                    target_pos = np.array([self.init_pos_x, self.init_pos_y, self.init_pos_z])
                    distance = np.linalg.norm(current_pos - target_pos)
                    
                    if distance < 0.1:  # 到达搜索位置
                        rospy.loginfo("Reached search position, searching for AprilTag...")
                        self.send_velocity_command(0, 0, 0)  # 悬停
                    else:
                        # 飞向搜索位置
                        direction = (target_pos - current_pos) / distance
                        speed = min(1.0, distance)  # 最大速度1m/s
                        self.send_velocity_command(direction[0] * speed, 
                                                 direction[1] * speed, 
                                                 direction[2] * speed)
                        
            elif self.mission_state == "OBSERVE":
                # 保持在观察高度悬停
                if self.current_position:
                    current_z = self.current_position['position'].z
                    vz = (self.observe_height - current_z) * 0.5  # 简单的高度控制
                    self.send_velocity_command(0, 0, vz)
                    
            elif self.mission_state == "WAIT_POSITION":
                # 飞向等待位置
                if self.target_wait_position and self.current_position:
                    current_pos = np.array([
                        self.current_position['position'].x,
                        self.current_position['position'].y,
                        self.current_position['position'].z
                    ])
                    target_pos = np.array(self.target_wait_position)
                    
                    error = target_pos - current_pos
                    distance = np.linalg.norm(error)
                    
                    if distance < 0.1:  # 到达等待位置，误差小于0.1m
                        rospy.loginfo(f"Reached wait position at {self.wait_at_position}, entering WAIT state")
                        self.mission_state = "WAIT"
                        self.wait_apriltag_positions = []  # 清空之前的记录
                        self.wait_start_time = rospy.Time.now()
                        self.send_velocity_command(0, 0, 0)
                    else:
                        # 飞向等待位置
                        direction = error / distance
                        speed = min(0.5, distance)
                        self.send_velocity_command(direction[0] * speed, 
                                                 direction[1] * speed, 
                                                 direction[2] * speed)
                        
            elif self.mission_state == "WAIT":
                if self.target_wait_position and self.current_position:
                    current_pos = np.array([
                        self.current_position['position'].x,
                        self.current_position['position'].y,
                        self.current_position['position'].z
                    ])
                    target_pos = np.array(self.target_wait_position)
                    
                    error = target_pos - current_pos
                    distance = np.linalg.norm(error)
                    
                    if distance > 0.1:
                        # 飞向等待位置
                        direction = error / distance
                        speed = min(0.5, distance)
                        self.send_velocity_command(direction[0] * speed, 
                                                 direction[1] * speed, 
                                                 direction[2] * speed)
                    else:
                        # 在等待位置悬停
                        self.send_velocity_command(0, 0, 0)
                        
            elif self.mission_state == "LAND":
                # 降落
                #self.send_velocity_command(0, 0, -self.landing_speed)
                
                if self.target_wait_position and self.current_position:
                    current_pos = np.array([
                        self.current_position['position'].x,
                        self.current_position['position'].y,
                        self.current_position['position'].z
                    ])
                    target_pos = np.array(self.target_wait_position)
                    
                    error = target_pos - current_pos
                    distance = np.linalg.norm(error)
                    
                    if distance > 0.1:
                        # 飞向等待位置
                        direction = error / distance
                        speed = max(0.5, distance)
                        self.send_velocity_command(direction[0] * speed, 
                                                direction[1] * speed, 
                                                -self.landing_speed)
                    else:
                        self.send_velocity_command(0, 0, -self.landing_speed)
                
                # 检查是否着陆
                if self.current_position and (self.current_position['position'].z - 0.2)  < 0.08:
                    response = self.command_client(self.cmd.command, self.cmd.param1,self.cmd.param2,0,0,0,0,0)
                    if response.success:
                        rospy.loginfo("Landing completed!")
                    self.send_velocity_command(0, 0, 0)
                    break
                    
            rate.sleep()
            
    def run(self):
        """运行主程序"""
        rospy.loginfo("Starting dynamic landing mission...")
        self.control_loop()

if __name__ == '__main__':
    try:
        controller = DynamicLandingController()
        controller.run()
    except rospy.ROSInterruptException:
        pass