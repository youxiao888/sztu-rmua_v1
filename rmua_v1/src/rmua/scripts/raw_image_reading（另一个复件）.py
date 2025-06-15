#!/usr/bin/env python3
import cv2
import rospy
import math
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Pose
from mavros_msgs.msg import State
from mavros_msgs.srv import SetMode
from mavros_msgs.msg import PositionTarget
from mavros_msgs.srv import CommandBool
import torch
import roslaunch
from ultralytics import YOLO
from collections import deque

class YoloV5Node:
    def __init__(self):
        rospy.init_node('yolov5_node', anonymous=True)

        # 存储历史位置和对应的时间戳
        self.position_history = deque(maxlen=10)  # 存储最近10个位置数据
        self.orientation_history = deque(maxlen=10)  # 存储最近10个朝向数据
        self.timestamp_history = deque(maxlen=10)  # 存储对应的时间戳

        self.current_goal = None
        self.goal_published = False

        self.target_points = []

        i = 0

        while rospy.has_param(f"~point{i}"):
            point_data = rospy.get_param(f"~point{i}")
            if "label" not in point_data:
                rospy.logerr(f"Point {i} does not have label data")
                break

            if "xyz" not in point_data:
                rospy.logerr(f"Point {i} does not have xyz data")
                break

            if len(point_data["xyz"]) != 3:
                rospy.logerr(f"Point {i} does not have 3 elements in xyz data")
                break

            process_point = {
                "label": point_data["label"],
                "xyz": point_data["xyz"],
            }

            if "yaw" in point_data:
                process_point["yaw"] = point_data["yaw"]

            if "length" in point_data:
                process_point["length"] = point_data["length"]

            self.target_points.append(process_point)

            i += 1

        model_path = rospy.get_param('~model_path')
        rospy.set_param('/CURRENT_FLAG', 0)
        self.CURRENT_FLAG = rospy.get_param('/CURRENT_FLAG', 0)
        
        self.depth_image = None
        self.color_image = None
        self.landing_image = None  # 存储降落相机图像
        self.cx = rospy.get_param('~cx', 654.9307250976562)
        self.cy = rospy.get_param('~cy', 365.23223876953125)
        self.fx = rospy.get_param('~fx', 913.9072265625)
        self.fy = rospy.get_param('~fy', 914.03125)
        self.landing_cx = rospy.get_param('~landing_cx', 160.0)
        self.landing_cy = rospy.get_param('~landing_cy', 120.0)
        self.landing_fx = rospy.get_param('~landing_fx', 277.191356)
        self.landing_fy = rospy.get_param('~landing_fy', 277.191356)
        self.sample_interval = rospy.get_param('~sample_interval', 5)
        self.k = rospy.get_param('~k', 1.0)
        self.max_iterations = rospy.get_param('~max_iterations', 4)
        self.depth_diff_threshold = rospy.get_param('~depth_diff_threshold', 0.1)
        self.depth_ends_diff_threshold = rospy.get_param('~depth_ends_diff_threshold', 0.25)
        self.max_bad_lines_ratio = rospy.get_param('~max_bad_lines_ratio', 0.50)
        self.aspect_ratio_min = rospy.get_param('~aspect_ratio_min', 0.95)
        self.aspect_ratio_max = rospy.get_param('~aspect_ratio_max', 1.05)
        self.iteration_multiple = rospy.get_param('~iteration_multiple', 0.05)
        self.sample_count = int(180 / self.sample_interval)
        self.landing_launch_file = rospy.get_param('~landing_launch_file', '/home/amov/rmua_v1/src/rmua/launch/landing0.launch')
        
        # 时间戳比对阈值
        self.timestamp_threshold = 0.05  # 50ms
        
        rospy.set_param("/IS_LANDING", False)
        
        # 旋转方框深度估算相关参数
        # 相机内参 (复用原有参数)
        self.camera_matrix = np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ])
        
        # 方框实际尺寸
        self.real_box_size = 1.4  # 米
        
        # 存储最新的图像和检测结果
        self.latest_color_image = None
        self.prev_detection = None
        self.frame_count = 0
        self.current_reference_width = None  # 当前有效的参考宽度
        
        # 存储最新的检测结果用于显示
        self.latest_detection_result = None
        self.latest_info_texts = ["Waiting for detection..."]
        
        # 正方形检测相关的时间控制
        self.last_square_detection_time = rospy.Time(0)
        self.square_detection_interval = rospy.Duration(0.5)  # 0.5秒间隔
        self.latest_square_result = None  # 存储最新的正方形检测结果
        
        # CUDA优化设置
        if torch.cuda.is_available():
            torch.cuda.set_device(0)  # 明确设置GPU设备
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'
            print("not using cuda")
            
        self.model = YOLO(model_path)
        
        self.min_valid_depth = 100
        self.max_valid_depth = 5000
        self.process_lock = False
        self.landing_process_lock = False  # 降落处理锁

        self.bridge = CvBridge()

        self.last_square_size = 0.0
        self.square_center = None 
        self.local_position = None
        self.local_orientation = None
        self.have_pub_circle_through = False
        self.current_flag = 0
        self.init_flag = False
        self.Land_command = False
        self.target_msg = PositionTarget()
        self.last_x_local = None
        self.last_y_local = None
        self.last_time = None
        self.yaw_rate = None

        self.mavros_state = None
        self.last_circle_position = None
        self.circle_stage = 0  # 0: 初始，1: 已发送前置点，2: 已发送穿过点
        self.last_setpoint_time = rospy.Time.now()
        
        self.body_vel = None
        self.haved_launched_landing_file = False
        self.odometry_timestamp = None
        
        # 订阅和发布
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.callback, queue_size=1, buff_size=2**24)
        self.depth_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback, queue_size=1, buff_size=2**24)
        self.local_position_sub = rospy.Subscriber("/mavros/local_position/odom", Odometry, self.local_position_callback)
        self.goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=10, latch=True)
        self.local_pos_pub = rospy.Publisher('/mavros/setpoint_position/local', PoseStamped, queue_size=10)
        self.mavros_state_sub = rospy.Subscriber('/mavros/state', State, self.mavros_state_callback)
        self.setpoint_pub = rospy.Publisher('/mavros/setpoint_raw/local', PositionTarget, queue_size=10)
        
        # 移除原来的detection_timer，改为在callback中控制时间间隔
        

        try:
            self.set_mode_client = rospy.ServiceProxy('/mavros/set_mode', SetMode)
            rospy.wait_for_service('/mavros/set_mode', timeout=5.0)
            rospy.loginfo("MAVROS服务已连接")
        except rospy.ROSException:
            rospy.logwarn("无法连接到MAVROS服务，将继续运行")

    def mavros_state_callback(self, state_msg):
        self.mavros_state = state_msg

    def calculate_distance(self, x, y, z):
        """统一使用前两个坐标计算距离"""
        if self.CURRENT_FLAG >= len(self.target_points):
            return float('inf')
        target_point = self.target_points[self.CURRENT_FLAG]["xyz"]
        return math.sqrt((x-target_point[0])**2 + (y-target_point[1])**2 + (z-target_point[2])**2)

    def depth_callback(self, data):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
            self.depth_timestamp = data.header.stamp.to_sec()
        except Exception as e:
            rospy.logerr(e)
            self.depth_image = None

    def local_position_callback(self, data):
        try:
            # 存储当前位置和时间戳
            current_position = data.pose.pose.position
            current_orientation = data.pose.pose.orientation
            timestamp = data.header.stamp.to_sec()
            
            # 添加到历史记录队列
            self.position_history.append(current_position)
            self.orientation_history.append(current_orientation)
            self.timestamp_history.append(timestamp)
            
            # 更新当前位置信息
            self.local_position = current_position
            self.local_orientation = current_orientation
            self.yaw_rate = data.twist.twist.angular.z
            self.body_vel = math.sqrt(data.twist.twist.linear.x**2 + data.twist.twist.linear.y**2 + data.twist.twist.linear.z**2)
            self.odometry_timestamp = timestamp

            if self.current_goal and self.goal_published:
                dx = self.local_position.x - self.current_goal[0]
                dy = self.local_position.y - self.current_goal[1]
                dz = self.local_position.z - self.current_goal[2]
                if math.sqrt(dx**2 + dy**2 + dz**2) < 1:

                    if self.target_points[self.CURRENT_FLAG]["label"] == "circle" or self.target_points[self.CURRENT_FLAG]["label"] == "waypoint" or self.target_points[self.CURRENT_FLAG]["label"] == "square":
                        self.CURRENT_FLAG = rospy.get_param('/CURRENT_FLAG', 0) + 1
                        rospy.set_param('/CURRENT_FLAG', self.CURRENT_FLAG)
                        self.goal_published = False
                        self.current_goal = None
                        rospy.loginfo(f"Reached goal, progressing to target {self.CURRENT_FLAG}")
                        
                    elif self.target_points[self.CURRENT_FLAG]["label"] == "special_circle":
                        self.CURRENT_FLAG = rospy.get_param('/CURRENT_FLAG', 0) + 1
                        rospy.set_param('/CURRENT_FLAG', self.CURRENT_FLAG)
                        self.goal_published = False
                        self.current_goal = None
                        rospy.loginfo(f"Reached goal, progressing to target {self.CURRENT_FLAG}")
                        
                    """
                    elif self.target_points[self.CURRENT_FLAG]["label"] == "special_circle":
                        if math.sqrt(dx**2 + dy**2 + dz**2) < 0.1:
                            rospy.loginfo("Circle traversal completed!")
                            rospy.set_param('/EGO_MODE', True)
                            rospy.set_param('/OFFBOARD_MODE', False)
                            self.CURRENT_FLAG = rospy.get_param('/CURRENT_FLAG', 0) + 1
                            rospy.set_param('/CURRENT_FLAG', self.CURRENT_FLAG)
                            self.circle_stage = 0
                            self.goal_published = False
                            self.current_goal = None
                    """
                            
            
            # 主动发布四元素航点
            if self.CURRENT_FLAG < len(self.target_points) and self.target_points[self.CURRENT_FLAG]["label"] == "waypoint":
                current_target = self.target_points[self.CURRENT_FLAG]["xyz"]
                if not self.goal_published:
                    x, y, z = current_target
                    self.publish_goal(x, y, z, self.target_points[self.CURRENT_FLAG]["yaw"])
                    self.current_goal = (x, y, z)
                    self.goal_published = True

            # 检查是否需要开始降落

                    
        except Exception as e:
            rospy.logerr(f"位置回调错误: {e}")

    def callback(self, data):
        
        self.color_timestamp = data.header.stamp.to_sec()
        
        if self.process_lock or self.depth_image is None or self.local_position is None or self.target_points[self.CURRENT_FLAG]["label"] == "Land":
            return
                
        self.process_lock = True
        
        try:
            if self.CURRENT_FLAG < len(self.target_points) and self.target_points[self.CURRENT_FLAG]["label"] == "waypoint":
                self.process_lock = False
                return
            
            # 解析图像
            try:
                cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
                # 为旋转方框检测更新最新图像
                self.latest_color_image = cv_image.copy()
            except Exception as e:
                rospy.logerr(e)
                self.process_lock = False
                return

            results = self.model.predict(
                cv_image,
                device='cuda:0',
                verbose=False
            )
                    
            # 处理不同目标类型
            if self.CURRENT_FLAG < len(self.target_points) and self.target_points[self.CURRENT_FLAG]["label"] == "square" and self.yaw_rate < 0.01 and self.body_vel < 0.1:
                # 检查是否需要执行正方形检测（每0.5秒一次）
                current_time = rospy.Time.now()
                if (current_time - self.last_square_detection_time) >= self.square_detection_interval:
                    self.last_square_detection_time = current_time
                    
                    # 使用旋转方框深度估算处理正方形目标
                    depth_meters, center_x, center_y = self.process_square_with_rotated_box(results, cv_image)
                    # 存储检测结果供实时显示使用
                    self.latest_square_result = (depth_meters, center_x, center_y)
                    
                    if depth_meters is not None and center_x is not None and center_y is not None:
                        # 使用检测到的中心作为像素点
                        pixel_point = (center_x, center_y)
                        
                        def pixel2cam(point, depth):
                            return ((point[0] - self.cx) * depth / self.fx,
                                    (point[1] - self.cy) * depth / self.fy, depth)
                        
                        # 计算像素点到相机坐标的转换
                        center_cam = pixel2cam(pixel_point, depth_meters)
                        body_coords = self.transform_camera_to_body(*center_cam)
                        x_actual_local, y_actual_local, z_actual_local = self.transform_body_to_local(*body_coords, self.color_timestamp)  
                        x_local, y_local, z_local = self.transform_body_to_local(body_coords[0] + 1, body_coords[1], body_coords[2], self.color_timestamp)
                        
                        # 检查是否获得了有效位置
                        if x_local is not None:
                            self.square_center = x_actual_local, y_actual_local, z_actual_local
                            rospy.loginfo(f"Square center at {self.square_center}")
                            
                            if self.CURRENT_FLAG < len(self.target_points):
                                target_distance = self.calculate_distance(x_actual_local, y_actual_local, z_actual_local)
                                if target_distance < 1 and not self.goal_published:
                                    self.publish_goal(x_local, y_local, z_local)
                                    rospy.loginfo(f"Published goal at ({x_local:.2f}, {y_local:.2f}, {z_local:.2f})")
                else:
                    # 如果不在检测时间窗口内，使用之前的检测结果进行显示
                    if self.latest_square_result is not None:
                        depth_meters, center_x, center_y = self.latest_square_result
                        if center_x is not None and center_y is not None:
                            # 在图像上标出中心点
                            cv2.circle(cv_image, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
                            
                            # 显示基本信息
                            cv2.putText(cv_image, f"Using cached result", (50, 50), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                            if depth_meters is not None:
                                cv2.putText(cv_image, f"Depth: {depth_meters:.2f}m", (50, 90), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            elif self.yaw_rate < 0.01 and self.CURRENT_FLAG < len(self.target_points) and self.target_points[self.CURRENT_FLAG]["label"] == "circle":
                
                depth_meters, center_x, center_y = self.process_general_circle(results, cv_image)
                
                
                # 处理圆形目标 - 使用内框中心点
                if depth_meters is not None:

                    pixel_point = (center_x, center_y)
                    # 计算像素到相机坐标的转换
                    x_cam = (pixel_point[0] - self.cx) * depth_meters / self.fx
                    y_cam = (pixel_point[1] - self.cy) * depth_meters / self.fy
                    z_cam = depth_meters

                    x_body, y_body, z_body = self.transform_camera_to_body(x_cam, y_cam, z_cam)
                    #print(f"x_body, y_body, z_body:{x_body, y_body, z_body}")
                    x_local, y_local, z_local, time_interval = self.transform_body_to_local(x_body + 1, y_body, z_body, self.color_timestamp)          
                    x_actual_local, y_actual_local, z_actual_local, time_interval = self.transform_body_to_local(x_body, y_body, z_body, self.color_timestamp) 
                    
                    if x_actual_local is not None:

                        target_distance = self.calculate_distance(x_actual_local, y_actual_local, z_actual_local)
                        #print(f"self.local_position.x,self.local_position.y,self.local_position.z{self.local_position.x,self.local_position.y,self.local_position.z}")
                        #print(f"x_actual_local, y_actual_local, z_actual_loca,time_interval:{x_actual_local, y_actual_local, z_actual_local,time_interval}")
                        if target_distance < 0.5 and not self.goal_published:
                            
                            self.publish_goal(x_local, y_local, z_local)
                            rospy.loginfo(f"Published goal at ({x_local:.2f}, {y_local:.2f}, {z_local:.2f})")
                            
            elif self.yaw_rate < 0.01 and self.CURRENT_FLAG < len(self.target_points) and  self.target_points[self.CURRENT_FLAG]["label"] == "special_circle" and self.body_vel < 0.1:
                
                depth_meters, center_x, center_y = self.process_general_circle(results, cv_image)
                
                
                # 处理圆形目标 - 使用内框中心点
                if depth_meters is not None:

                    pixel_point = (center_x, center_y)
                    # 计算像素到相机坐标的转换
                    x_cam = (pixel_point[0] - self.cx) * depth_meters / self.fx
                    y_cam = (pixel_point[1] - self.cy) * depth_meters / self.fy
                    z_cam = depth_meters

                    x_body, y_body, z_body = self.transform_camera_to_body(x_cam, y_cam, z_cam)
                    #print(f"x_body, y_body, z_body:{x_body, y_body, z_body}")
                    x_local, y_local, z_local, time_interval = self.transform_body_to_local(x_body + 0.5, y_body, z_body + 0.15, self.color_timestamp)          
                    x_actual_local, y_actual_local, z_actual_local, time_interval = self.transform_body_to_local(x_body, y_body, z_body, self.color_timestamp) 
                    
                    if x_actual_local is not None:

                        target_distance = self.calculate_distance(x_actual_local, y_actual_local, z_actual_local)
                        #print(f"self.local_position.x,self.local_position.y,self.local_position.z{self.local_position.x,self.local_position.y,self.local_position.z}")
                        #print(f"x_actual_local, y_actual_local, z_actual_loca,time_interval:{x_actual_local, y_actual_local, z_actual_local,time_interval}")
                        if target_distance < 0.5 and not self.goal_published:
                            
                            self.publish_goal(x_local, y_local, z_local)
                            rospy.loginfo(f"Published goal at ({x_local:.2f}, {y_local:.2f}, {z_local:.2f})")
                    
            # 显示结果图像
            cv2.imshow("YOLO Inference", cv_image)
            cv2.waitKey(1)
                
        except Exception as e:
            rospy.logerr(f"处理图像时发生错误: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
        finally:
            self.process_lock = False

    def process_square_with_rotated_box(self, results, cv_image):
        """使用旋转方框深度估算处理正方形目标"""
        try:
            # 从YOLO检测结果中提取旋转框信息
            box_points, confidence = self.get_rotated_box_info(results[0])
            
            if box_points is None:
                cv2.putText(cv_image, "No valid detection box", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                return None, None, None
            
            # 计算框的尺寸信息
            center_x, center_y, width, height, aspect_ratio = self.calculate_box_dimensions(box_points)
            
            if center_x is None:
                cv2.putText(cv_image, "Invalid box dimensions", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                return None, None, None
            
            current_detection = {
                'center_x': center_x,
                'center_y': center_y,
                'width': width,
                'height': height,
                'aspect_ratio': aspect_ratio,
                'box_points': box_points,
                'confidence': confidence,
                'frame': self.frame_count
            }
            
            depth_meters = None
            

            if self.prev_detection is None:
                self.prev_detection = current_detection
                
                # 在图像上显示第一帧信息
                cv2.putText(cv_image, "** FIRST FRAME - NO DEPTH **", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                cv2.putText(cv_image, f"Recording reference width: {width:.1f}px", (50, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
                rospy.loginfo(f"帧{self.frame_count}: 第一帧检测 - 宽度={width:.1f}px, "
                            f"长宽比={aspect_ratio:.2f}, 不计算深度")
            else:
                # 判断是否满足更新条件
                update_reference = self.should_update_reference(self.prev_detection, current_detection)
                
                # 只有满足条件时才计算深度
                if update_reference:
                    self.current_reference_width = width
                    depth_meters = self.estimate_depth_from_box_size(width)
                    
                    rospy.loginfo(f"帧{self.frame_count}: 更新参考框 - 宽度={width:.1f}px, "
                                f"长宽比={aspect_ratio:.2f}, 估算深度={depth_meters:.2f}m")
                    
                    # 更新前一帧检测结果
                    self.prev_detection = current_detection
                    
                    # 在图像上显示更新信息
                    cv2.putText(cv_image, f"** REFERENCE UPDATED **", (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(cv_image, f"Depth: {depth_meters:.2f}m", (50, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    # 不满足条件，显示原因
                    reasons = []
                    if width <= self.prev_detection['width']:
                        reasons.append("width not increased")
                    if not (0.8 < aspect_ratio < 1.2):
                        reasons.append("aspect ratio invalid")
                    
                    if reasons:
                        cv2.putText(cv_image, f"Conditions not met: {', '.join(reasons)}", (50, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                    
                    # 如果有有效的参考宽度，可以显示当前状态
                    if self.current_reference_width is not None:
                        depth_meters = self.estimate_depth_from_box_size(self.current_reference_width)
                        cv2.putText(cv_image, f"Using Ref Width: {self.current_reference_width:.1f}px", (50, 90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                        cv2.putText(cv_image, f"Depth: {depth_meters:.2f}m", (50, 130), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    rospy.loginfo(f"帧{self.frame_count}: 检测到目标但不满足更新条件 - "
                                f"宽度={width:.1f}px, 长宽比={aspect_ratio:.2f}")
                
                # 始终更新prev_detection用于下一帧比较
                self.prev_detection.update({
                    'width': width,
                    'height': height,
                    'aspect_ratio': aspect_ratio,
                    'confidence': confidence,
                    'frame': self.frame_count
                })
            
            # 绘制检测框和信息
            box_points_int = box_points.astype(int)
            cv2.polylines(cv_image, [box_points_int], True, (0, 255, 0), 2)
            
            # 绘制中心点
            cv2.circle(cv_image, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
            
            # 显示基本信息
            cv2.putText(cv_image, f"Width: {width:.1f}px", (50, 170), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(cv_image, f"Aspect Ratio: {aspect_ratio:.2f}", (50, 210), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(cv_image, f"Confidence: {confidence:.2f}", (50, 250), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(cv_image, f"Frame: {self.frame_count}", (50, 290), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            self.frame_count += 1
            
            return depth_meters, center_x, center_y
            
        except Exception as e:
            rospy.logerr(f"旋转方框处理错误: {e}")
            return None, None, None

    def get_rotated_box_info(self, detection_result):
        """从YOLO检测结果中提取旋转框信息"""
        try:
            # 获取旋转框信息 (假设YOLO返回的是OBB格式)
            if hasattr(detection_result, 'obb') and detection_result.obb is not None:
                obb = detection_result.obb
                boxes = obb.xyxyxyxy.cpu().numpy()  # 8个坐标点 (4个角点的x,y坐标)
                confidences = obb.conf.cpu().numpy()
                
                if len(boxes) > 0:
                    # 选择置信度最高的框
                    max_conf_idx = np.argmax(confidences)
                    box_points = boxes[max_conf_idx].reshape(4, 2)
                    confidence = confidences[max_conf_idx]
                    
                    return box_points, confidence
            
            # 如果没有OBB，尝试使用常规边界框
            elif hasattr(detection_result, 'boxes') and detection_result.boxes is not None:
                boxes = detection_result.boxes.xyxy.cpu().numpy()
                confidences = detection_result.boxes.conf.cpu().numpy()
                
                if len(boxes) > 0:
                    # 选择面积最大的框
                    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
                    max_area_idx = np.argmax(areas)
                    box = boxes[max_area_idx]
                    confidence = confidences[max_area_idx]
                    
                    # 将矩形框转换为4个角点
                    x1, y1, x2, y2 = box
                    box_points = np.array([
                        [x1, y1], [x2, y1], [x2, y2], [x1, y2]
                    ])
                    
                    return box_points, confidence
                    
        except Exception as e:
            rospy.logwarn(f"提取旋转框信息错误: {e}")
        
        return None, 0

    def calculate_box_dimensions(self, box_points):
        """计算旋转框的宽度、高度和中心"""
        if box_points is None or len(box_points) != 4:
            return None, None, None, None, None
        
        # 计算中心点
        center_x = np.mean(box_points[:, 0])
        center_y = np.mean(box_points[:, 1])
        
        # 计算各边的长度
        side_lengths = []
        for i in range(4):
            p1 = box_points[i]
            p2 = box_points[(i + 1) % 4]
            length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            side_lengths.append(length)
        
        # 假设相邻的两边分别是宽和高
        width = side_lengths[0]
        height = side_lengths[1]
        
        # 计算长宽比
        aspect_ratio = height / width if height > 0 else 0
        
        return center_x, center_y, width, height, aspect_ratio

    def estimate_depth_from_box_size(self, box_width_pixels):
        """根据框的像素宽度估算深度"""
        if box_width_pixels <= 0:
            return 0
        
        # 使用相似三角形原理: depth = (real_size * focal_length) / pixel_size
        estimated_depth = (self.real_box_size * self.fx) / box_width_pixels
        return estimated_depth

    def should_update_reference(self, prev_detection, current_detection):
        """判断是否应该更新参考框"""
        if prev_detection is None:
            return True
        
        prev_width = prev_detection['width']
        current_width = current_detection['width']
        current_aspect_ratio = current_detection['aspect_ratio']
        
        # 检查条件：当前帧宽度更大，且长宽比在0.8-1.2之间
        width_increased = current_width > (prev_width + 0.1 * prev_width)
        aspect_ratio_valid = 0.8 < current_aspect_ratio < 1.2
        
        return width_increased and aspect_ratio_valid
            
    def process_general_circle(self, results, cv_image):
        
        boxes = results[0].boxes
        if len(boxes) == 0:
            self.status_details = "No circle detected"
            cv2.putText(cv_image, self.status_details, (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.waitKey(1)
            return None, None, None
        
        if len(boxes) > 1:
            # 计算每个框的面积
            areas = []
            for box in boxes.xyxy.cpu().numpy():
                area = (box[2] - box[0]) * (box[3] - box[1])
                areas.append(area)
            
            # 找到最大面积的框
            max_idx = np.argmax(areas)
            box = boxes.xyxy.cpu().numpy()[max_idx]
        else:
            box = boxes.xyxy.cpu().numpy()[0]
            
                # 计算检测框的宽度和高度
        box_width = box[2] - box[0]
        box_height = box[3] - box[1]
        
        aspect_ratio = box_width / box_height if box_height > 0 else 0
        
        # 检查长宽比是否在允许范围内
        if aspect_ratio < self.aspect_ratio_min or aspect_ratio > self.aspect_ratio_max:

            self.status_details = f"Invalid aspect ratio: {aspect_ratio:.2f} (should be between {self.aspect_ratio_min} and {self.aspect_ratio_max})"
            cv2.putText(cv_image, self.status_details, (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(cv_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
            return None, None, None
        
        cx = int((box[0] + box[2]) / 2)
        cy = int((box[1] + box[3]) / 2)
        
        cv2.circle(cv_image, (cx, cy), 5, (0, 255, 0), -1)
        
        box_size = min(box_width, box_height)
        initial_line_length = box_size
        step_size = initial_line_length * self.iteration_multiple
        
        current_length = initial_line_length
        prev_depth = None
        valid_depth = None
        valid_sample_points = None
        valid_line_length = None
        valid_paired_samples = None
        valid_bad_lines_count = None
        valid_bad_lines_ratio = None
        
        iteration_failures = []
        
        for iteration in range(self.max_iterations):
            # 采样当前线段长度的深度
            all_result, paired_samples, sample_points = self.sample_depth_for_line_length(cx, cy, current_length)
            
            # 检查是否获取到了足够的样本点
            if len(paired_samples) == 0:
                iteration_failures.append(f"Iteration {iteration+1}: No valid sample pairs")
                current_length -= step_size
                continue
                
            # 检查是否获取到了有效的深度
            all_depth, all_filtered = all_result
            if all_depth is None:
                iteration_failures.append(f"Iteration {iteration+1}: No valid filtered depth")
                current_length -= step_size
                continue
            
            # 计算有多少线段的两端深度差大于阈值
            bad_lines_count = 0
            for inner_depth, outer_depth in paired_samples:
                if abs(inner_depth - outer_depth) > self.depth_ends_diff_threshold:
                    bad_lines_count += 1
            
            # 计算不合格线段的比例
            bad_lines_ratio = bad_lines_count / len(paired_samples)
            
            # 检查是否满足条件
            depth_diff_ok = prev_depth is None or abs(all_depth - prev_depth) < self.depth_diff_threshold
            ends_diff_ok = bad_lines_ratio <= self.max_bad_lines_ratio
            
            # 记录不满足条件的原因
            if not depth_diff_ok:
                iteration_failures.append(f"Iteration {iteration+1}: Depth difference too large: {abs(all_depth - prev_depth):.3f}m > {self.depth_diff_threshold}m")
            
            if not ends_diff_ok:
                iteration_failures.append(f"Iteration {iteration+1}: Too many bad lines: {bad_lines_count}/{len(paired_samples)} ({bad_lines_ratio*100:.1f}%) > {self.max_bad_lines_ratio*100}%")
            
            if depth_diff_ok and ends_diff_ok:
                valid_depth = all_depth
                valid_sample_points = sample_points
                valid_line_length = current_length
                valid_paired_samples = paired_samples
                valid_bad_lines_count = bad_lines_count
                valid_bad_lines_ratio = bad_lines_ratio
                break
            
            # 记录当前深度用于下次比较
            prev_depth = all_depth
            
            # 减小线段长度
            current_length -= step_size
            
        if valid_depth is None:

            self.status_details = "Failed to find valid depth after iterations"
            
            for i, failure in enumerate(iteration_failures[-min(2, len(iteration_failures)):]):
                cv2.putText(cv_image, failure, (50, 90 + i*40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv2.putText(cv_image, self.status_details, (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(cv_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 165, 0), 2)
            return None, None, None
        
        for x, y, point_type in valid_sample_points:
            color = (255, 0, 0) if point_type == 'inner' else (0, 0, 255)
            cv2.circle(cv_image, (x, y), 3, color, -1)
            
        bad_line_angles = []
        for i in range(0, len(valid_sample_points), 2):
            if i+1 < len(valid_sample_points):
                inner_point = valid_sample_points[i]
                outer_point = valid_sample_points[i+1]
                inner_x, inner_y, _ = inner_point
                outer_x, outer_y, _ = outer_point
                
                # 获取对应的深度差
                inner_depth = valid_paired_samples[i//2][0]
                outer_depth = valid_paired_samples[i//2][1]
                depth_diff = abs(inner_depth - outer_depth)
                
                # 如果深度差大于阈值，用红色绘制线段，否则用绿色
                is_bad_line = depth_diff > self.depth_ends_diff_threshold
                line_color = (0, 0, 255) if is_bad_line else (0, 255, 0)
                cv2.line(cv_image, (inner_x, inner_y), (outer_x, outer_y), line_color, 1)
                
                # 如果是不合格线段，记录角度
                if is_bad_line:
                    angle = math.atan2(outer_y - inner_y, outer_x - inner_x) * 180 / math.pi
                    bad_line_angles.append(angle)
        
        # 在图像上显示结果
        cv2.putText(cv_image, f"Depth: {valid_depth:.3f}m", (50, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(cv_image, f"Bad lines: {valid_bad_lines_count}/{len(valid_paired_samples)} ({valid_bad_lines_ratio*100:.1f}%)", 
                    (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(cv_image, f"Line length: {valid_line_length:.1f}px", (50, 170), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(cv_image, f"Aspect ratio: {aspect_ratio:.2f}", (50, 210), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 如果有不合格线段，显示它们的角度
        if bad_line_angles:
            bad_angles_text = "Bad line angles: " + ", ".join([f"{angle:.0f}°" for angle in bad_line_angles[:3]])
            if len(bad_line_angles) > 3:
                bad_angles_text += f" and {len(bad_line_angles) - 3} more"
            cv2.putText(cv_image, bad_angles_text, (50, 250), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # 绘制检测框
        cv2.rectangle(cv_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 165, 0), 2)
        
        return valid_depth, cx, cy
            
    def sample_depth_for_line_length(self, cx, cy, line_length):
        """采样给定线段长度的深度值"""
        half_length = line_length / 2

        all_samples = []  # 所有采样的深度值
        paired_samples = []  # 成对的内外采样点深度值
        sample_points = []  # 采样点坐标
        
        for i in range(self.sample_count):
            angle = i * self.sample_interval * math.pi / 180.0
            
            # 计算方向向量
            dx = math.sin(angle)
            dy = -math.cos(angle)  # 负号是因为图像坐标系y轴向下
            
            # 计算内侧采样点
            inner_x = int(cx - half_length * dx)
            inner_y = int(cy - half_length * dy)
            
            # 计算外侧采样点
            outer_x = int(cx + half_length * dx)
            outer_y = int(cy + half_length * dy)
            
            # 获取内侧点的深度
            inner_depth = self.get_valid_depth(inner_x, inner_y)
            
            # 获取外侧点的深度
            outer_depth = self.get_valid_depth(outer_x, outer_y)
            
            # 只有当内外两端点都有有效深度时，才记录这条线段
            if inner_depth is not None and outer_depth is not None:
                paired_samples.append((inner_depth, outer_depth))
                
                # 只有当线段的两端深度差小于阈值才参加计算所有采样点的平均深度
                if abs(inner_depth - outer_depth) <= self.depth_ends_diff_threshold:
                    all_samples.extend([inner_depth, outer_depth])
                    
                sample_points.append((inner_x, inner_y, 'inner'))
                sample_points.append((outer_x, outer_y, 'outer'))
        
        # 计算所有采样点的平均深度
        all_result = self.get_filtered_depth(all_samples)
        
        return all_result, paired_samples, sample_points
    
    def get_filtered_depth(self, samples):
        """使用四分位距法过滤异常值并计算平均深度"""
        if len(samples) == 0:
            return None, []
            
        samples = np.array(samples)
        Q1 = np.percentile(samples, 25)
        Q3 = np.percentile(samples, 75)
        IQR = Q3 - Q1
        
        # 定义异常值边界
        lower_bound = Q1 - self.k * IQR
        upper_bound = Q3 + self.k * IQR
        
        # 过滤异常值
        mask = (samples >= lower_bound) & (samples <= upper_bound)
        filtered = samples[mask]
        
        if len(filtered) == 0:
            return None, []
            
        return np.mean(filtered), filtered
    
    def get_valid_depth(self, x, y):
        """获取有效的深度值，处理异常"""
        if not (0 <= x < self.depth_image.shape[1] and 0 <= y < self.depth_image.shape[0]):
            return None
            
        depth = self.depth_image[y, x]
        
        # 处理无效深度值
        if depth == 0 or np.isnan(depth):
            # 在周围3x3区域内寻找有效深度
            min_y = max(0, y-1)
            max_y = min(self.depth_image.shape[0], y+2)
            min_x = max(0, x-1)
            max_x = min(self.depth_image.shape[1], x+2)
            
            depth_window = self.depth_image[min_y:max_y, min_x:max_x]
            valid_depths = depth_window[depth_window > 0]
            
            if len(valid_depths) > 0:
                depth = np.median(valid_depths)
            else:
                return None
        
        # 转换单位：毫米到米
        return depth / 1000.0
            
    def handle_circle_flight(self, x_body, y_body, z_body, x_actual_local, y_actual_local, z_actual_local):
        """处理CURRENT_FLAG=5，cls=17的特殊飞行逻辑"""
        print("Handling circle flight")
        # 计算圆前方位置（距离圆前方2米）
        x_front_body, y_front_body, z_front_body = x_body - 2, y_body, z_body + 0.1
        x_front_local, y_front_local, z_front_local, time_interval = self.transform_body_to_local(x_front_body, y_front_body, z_front_body, self.color_timestamp)
        
        # 检查是否获得了有效位置
        if x_front_local is None:
            return
        
        # 计算圆后方位置（距离圆后方1米）
        x_through_body, y_through_body, z_through_body = x_body + 1, y_body, z_body +0.1
        x_through_local, y_through_local, z_through_local, time_interval = self.transform_body_to_local(x_through_body, y_through_body, z_through_body, self.color_timestamp)
        
        # 检查是否获得了有效位置
        if x_through_local is None:
            return
        
        current_pos = (self.local_position.x, self.local_position.y, self.local_position.z)
        current_circle_pos = (x_actual_local, y_actual_local, z_actual_local)
        
        # 检查是否需要设置为offboard模式
        if self.mavros_state and self.mavros_state.mode != "OFFBOARD" and self.circle_stage >= 1:
            try:
                response = self.set_mode_client(custom_mode="OFFBOARD")
                if response.mode_sent:
                    rospy.loginfo("OFFBOARD mode enabled")
                    rospy.set_param('/OFFBOARD_MODE', True)
                else:
                    rospy.logwarn("Failed to set OFFBOARD mode")
            except rospy.ServiceException as e:
                rospy.logerr(f"Service call failed: {e}")
        
        # 如果是初始阶段，发送前置点
        if self.circle_stage == 0:
            # 发送前置点
            self.current_goal = (x_front_local, y_front_local, z_front_local)
            self.publish_setpoint(x_front_local, y_front_local, z_front_local)
            rospy.loginfo(f"Published front position: ({x_front_local:.2f}, {y_front_local:.2f}, {z_front_local:.2f})")
            self.last_circle_position = current_circle_pos
            self.circle_stage = 1
            self.current_goal = (x_front_local, y_front_local, z_front_local)
            
        # 如果已经发送了前置点，检查是否已到达前置点
        elif self.circle_stage == 1:
            dist_to_front = math.sqrt((current_pos[0] - x_front_local)**2 + 
                                     (current_pos[1] - y_front_local)**2 + 
                                     (current_pos[2] - z_front_local)**2)
            
            # 计算当前圆位置与上次记录的圆位置的差异
            if self.last_circle_position:
                pos_diff = math.sqrt((current_circle_pos[0] - self.last_circle_position[0])**2 + 
                                    (current_circle_pos[1] - self.last_circle_position[1])**2 + 
                                    (current_circle_pos[2] - self.last_circle_position[2])**2)
                

                self.last_circle_position = current_circle_pos
                # 如果已达前置点且圆位置稳定，发送穿过点
                if dist_to_front < 0.2 and pos_diff < 0.2:
                    self.current_goal = (x_through_local, y_through_local, z_through_local)
                    self.publish_setpoint(x_through_local, y_through_local, z_through_local)
                    rospy.loginfo(f"Published through position: ({x_through_local:.2f}, {y_through_local:.2f}, {z_through_local:.2f})")
                    self.circle_stage = 2
                    self.current_goal = (x_through_local, y_through_local,z_through_local)
                    self.goal_published = True
                else:
                    # 否则继续更新前置点位置
                    self.current_goal = (x_front_local, y_front_local, z_front_local)
                    self.publish_setpoint(x_front_local, y_front_local, z_front_local)
                    rospy.loginfo(f"Updated front position: ({x_front_local:.2f}, {y_front_local:.2f}, {z_front_local:.2f})")
                    rospy.loginfo(f"pos_diff: {pos_diff:.2f}, dist_to_front: {dist_to_front:.2f}")
            else:
                self.last_circle_position = current_circle_pos
                self.current_goal = (x_front_local, y_front_local, z_front_local)
                self.publish_setpoint(x_front_local, y_front_local, z_front_local)
                
        # 如果已经发送了穿过点，检查是否已穿过圆（前置阶段、穿过阶段）
        elif self.circle_stage == 2:
            dist_to_through = math.sqrt((current_pos[0] - x_through_local)**2 + 
                                       (current_pos[1] - y_through_local)**2 + 
                                       (current_pos[2] - z_through_local)**2)
            

            if self.have_pub_circle_through == False:
                self.current_goal = (x_through_local, y_through_local, z_through_local)

            self.publish_setpoint(x_through_local, y_through_local, z_through_local)
            self.have_pub_circle_through = True
    
    def publish_setpoint(self, x, y, z, yaw=0.0):
        """以10Hz频率发布本地位置设定点"""
        now = rospy.Time.now()
        if (now - self.last_setpoint_time).to_sec() >= 0.1:  # 10Hz
            pose_msg = PoseStamped()
            pose_msg.header.stamp = now
            pose_msg.header.frame_id = "map"
            pose_msg.pose.position.x = x
            pose_msg.pose.position.y = y
            pose_msg.pose.position.z = z
            
            if yaw != 0.0:
                q = quaternion_from_euler(0, 0, yaw)
                pose_msg.pose.orientation.x = q[0]
                pose_msg.pose.orientation.y = q[1]
                pose_msg.pose.orientation.z = q[2]
                pose_msg.pose.orientation.w = q[3]
            else:
                # 保持当前朝向
                pose_msg.pose.orientation = self.local_orientation
            
            self.local_pos_pub.publish(pose_msg)
            rospy.loginfo(f"Published setpoint at ({x:.2f}, {y:.2f}, {z:.2f},{yaw:.2f})" )
            self.last_setpoint_time = now

    def publish_goal(self, x, y, z, yaw=0.0):
        """通用航点发布函数"""
        goal_msg = PoseStamped()
        goal_msg.header.stamp = rospy.Time.now()
        goal_msg.header.frame_id = "map"
        goal_msg.pose.position.x = x
        goal_msg.pose.position.y = y
        goal_msg.pose.position.z = z
        
        if yaw != 0.0:
            q = quaternion_from_euler(0, 0, yaw)
            goal_msg.pose.orientation.x = q[0]
            goal_msg.pose.orientation.y = q[1]
            goal_msg.pose.orientation.z = q[2]
            goal_msg.pose.orientation.w = q[3]
        
        self.goal_pub.publish(goal_msg)
        self.current_goal = (x, y, z)
        self.goal_published = True
        rospy.loginfo(f"Published goal at ({x:.2f}, {y:.2f}, {z:.2f})" + 
                     f" with yaw {yaw:.2f}" if yaw != 0.0 else "")

    def transform_camera_to_body(self, x_cam, y_cam, z_cam):
        return z_cam, -x_cam, -y_cam

    def transform_body_to_local(self, x_body, y_body, z_body, frame_timestamp):
        """
        将机体坐标转换到局部坐标，使用时间戳匹配最近的位置数据
        """
        if not self.timestamp_history:
            rospy.logwarn("No position history available")
            return None, None, None, None
            
        # 查找与图像时间戳最接近的位置数据
        closest_idx = None
        min_time_diff = float('inf')
        
        for i, timestamp in enumerate(self.timestamp_history):
            time_diff = abs(timestamp - frame_timestamp)
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_idx = i
        
        # 判断时间差是否超过阈值
        if min_time_diff > self.timestamp_threshold:
            return None, None, None, None
            
        # 使用最接近的位置和朝向数据
        position = self.position_history[closest_idx]
        orientation = self.orientation_history[closest_idx]
        
        q = (orientation.x, orientation.y, orientation.z, orientation.w)
        yaw, pitch, roll = euler_from_quaternion(q, axes='szyx')

        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw), np.cos(yaw), 0],
                      [0, 0, 1]])
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                      [0, 1, 0],
                      [-np.sin(pitch), 0, np.cos(pitch)]])
        Rx = np.array([[1, 0, 0],
                      [0, np.cos(roll), -np.sin(roll)],
                      [0, np.sin(roll), np.cos(roll)]])
        R = Rz @ Ry @ Rx

        body_coords = np.array([x_body, y_body, z_body])
        local_coords = R @ body_coords

        return (local_coords[0] + position.x,
                local_coords[1] + position.y,
                local_coords[2] + position.z,
                min_time_diff)

    def launch_landing_file(self, launch_file_path):
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        launch = roslaunch.parent.ROSLaunchParent(uuid, [launch_file_path])
        launch.start()
        rospy.loginfo(f"Launch file '{launch_file_path}' started.")
        return launch

    def run(self):
        rate = rospy.Rate(30)  # 30Hz for main loop
        while not rospy.is_shutdown():
            if (self.target_points[self.CURRENT_FLAG]["label"] == "special_circle" and self.circle_stage > 0):
                # 在主循环中，确保setpoint以10Hz持续发布
                if self.circle_stage == 1 and self.current_goal:
                    rospy.loginfo(f"Current goal: {self.current_goal}")
                    self.publish_setpoint(self.current_goal[0], self.current_goal[1], 
                                         self.current_goal[2] if len(self.current_goal) > 2 else self.local_position.z)
                elif self.circle_stage == 2 and self.current_goal:
                    self.publish_setpoint(self.current_goal[0], self.current_goal[1], 
                                         self.current_goal[2] if len(self.current_goal) > 2 else self.local_position.z)
            if self.CURRENT_FLAG < len(self.target_points) and \
            self.target_points[self.CURRENT_FLAG]["label"] == "Land" and \
            not self.landing_process_lock and \
            self.haved_launched_landing_file == False:
                if self.body_vel is not None and self.body_vel < 0.1:

                    rospy.set_param('/EGO_MODE', False)
                    rospy.set_param("/IS_LANDING", True)                  
                    self.landing_process_lock = True
                    
                    self.launch_landing_file(self.landing_launch_file)
                    self.haved_launched_landing_file = True
                    print("send land command")
            rate.sleep()

if __name__ == '__main__':
    try:
        node = YoloV5Node()
        node.run()
    except Exception as e:
        rospy.logerr(f"Node exception: {e}")