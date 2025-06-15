#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import math

class RotatedBoxDepthEstimation:
    def __init__(self):
        rospy.init_node('rotated_box_depth_estimation', anonymous=True)
        
        # 初始化
        self.bridge = CvBridge()
        self.model = YOLO('/home/amov/Downloads/yolo11n-lamp-11.0x-finetune.pt')
        
        # 相机内参 (需要根据实际相机调整)
        self.camera_matrix = np.array([
            [927.345418, 0.0, 643.669443],
            [0.0, 922.006773, 369.176868],
            [0.0, 0.0, 1.0]
        ])
        self.fx = self.camera_matrix[0, 0]  # 焦距x
        self.fy = self.camera_matrix[1, 1]  # 焦距y
        self.cx = self.camera_matrix[0, 2]  # 主点x
        self.cy = self.camera_matrix[1, 2]  # 主点y
        
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
        
        # 订阅话题
        self.color_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.color_callback)
        self.depth_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_callback)
        
        # 检测处理定时器 - 每0.5秒处理一次
        self.detection_timer = rospy.Timer(rospy.Duration(0.5), self.process_detection)
        
        # 显示定时器 - 实时显示图像（约30fps）
        self.display_timer = rospy.Timer(rospy.Duration(0.033), self.display_image)
        
        rospy.loginfo("旋转方框深度估算节点已启动")
        rospy.loginfo(f"方框实际尺寸: {self.real_box_size}m")
        rospy.loginfo("检测频率: 每0.5秒一次, 显示频率: 实时")

    def color_callback(self, msg):
        try:
            self.latest_color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"彩色图像转换错误: {e}")

    def depth_callback(self, msg):
        # 保留深度话题订阅但不使用数据
        pass

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

    def process_detection(self, event):
        """每0.5秒执行一次检测处理"""
        if self.latest_color_image is None:
            return
        
        try:
            # YOLO预测
            results = self.model(self.latest_color_image)
            
            # 基本信息文本
            info_texts = [
                f"Frame: {self.frame_count}",
                f"Detection Rate: 0.5s",
                f"Display: Real-time",
                f"Status: {'Detecting...' if len(results) > 0 else 'No Detection'}"
            ]
            
            detection_result = None
            
            if len(results) > 0:
                # 提取旋转框信息
                box_points, confidence = self.get_rotated_box_info(results[0])
                
                if box_points is not None:
                    # 计算框的尺寸信息
                    center_x, center_y, width, height, aspect_ratio = self.calculate_box_dimensions(box_points)
                    
                    if center_x is not None:
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
                        
                        # 判断是否满足更新条件
                        update_reference = self.should_update_reference(self.prev_detection, current_detection)
                        
                        # 基本检测信息
                        info_texts = [
                            f"Frame: {self.frame_count}",
                            f"Detection Rate: 0.5s",
                            f"Display: Real-time",
                            f"Status: Detected",
                            f"Confidence: {confidence:.2f}",
                            f"Width: {width:.1f}px",
                            f"Height: {height:.1f}px",
                            f"Aspect Ratio: {aspect_ratio:.2f}"
                        ]
                        
                        # 只有满足条件时才显示深度相关信息
                        if update_reference:
                            self.current_reference_width = width
                            estimated_depth = self.estimate_depth_from_box_size(width)
                            
                            info_texts.extend([
                                f"Ref Width: {width:.1f}px",
                                f"Est Depth: {estimated_depth:.2f}m",
                                "** REFERENCE UPDATED **"
                            ])
                            
                            rospy.loginfo(f"帧{self.frame_count}: 更新参考框 - 宽度={width:.1f}px, "
                                        f"长宽比={aspect_ratio:.2f}, 估算深度={estimated_depth:.2f}m")
                            
                            # 更新前一帧检测结果
                            self.prev_detection = current_detection
                        else:
                            # 不满足条件，显示原因
                            reasons = []
                            if self.prev_detection is not None:
                                if width <= self.prev_detection['width']:
                                    reasons.append("width not increased")
                                if not (0.8 < aspect_ratio < 1.2):
                                    reasons.append("aspect ratio invalid")
                            
                            if reasons:
                                info_texts.append(f"Conditions not met: {', '.join(reasons)}")
                            
                            # 如果有有效的参考宽度，可以显示当前状态
                            if self.current_reference_width is not None:
                                info_texts.append(f"Current Ref Width: {self.current_reference_width:.1f}px")
                            
                            rospy.loginfo(f"帧{self.frame_count}: 检测到目标但不满足更新条件 - "
                                        f"宽度={width:.1f}px, 长宽比={aspect_ratio:.2f}")
                        
                        # 始终更新prev_detection用于下一帧比较
                        if self.prev_detection is None:
                            self.prev_detection = current_detection
                        else:
                            # 只更新基本信息，不更新参考状态
                            self.prev_detection.update({
                                'width': width,
                                'height': height,
                                'aspect_ratio': aspect_ratio,
                                'confidence': confidence,
                                'frame': self.frame_count
                            })
                        
                        detection_result = current_detection
                    else:
                        info_texts.append("Invalid box dimensions")
                else:
                    info_texts.append("No valid detection box")
            else:
                rospy.loginfo(f"帧{self.frame_count}: 未检测到目标")
            
            # 更新显示用的检测结果和信息
            self.latest_detection_result = detection_result
            self.latest_info_texts = info_texts
            self.frame_count += 1
            
        except Exception as e:
            rospy.logerr(f"检测处理错误: {e}")
            self.latest_info_texts = [f"Detection Error: {str(e)}"]

    def display_image(self, event):
        """实时显示图像"""
        if self.latest_color_image is None:
            return
        
        try:
            # 复制当前图像
            display_image = self.latest_color_image.copy()
            
            # 如果有检测结果，绘制检测框
            if self.latest_detection_result is not None:
                detection = self.latest_detection_result
                
                # 绘制旋转框
                box_points_int = detection['box_points'].astype(int)
                cv2.polylines(display_image, [box_points_int], True, (0, 255, 0), 2)
                
                # 绘制中心点
                center_x = int(detection['center_x'])
                center_y = int(detection['center_y'])
                cv2.circle(display_image, (center_x, center_y), 5, (0, 0, 255), -1)
            
            # 在图像上显示信息文本
            y_offset = 30
            for i, text in enumerate(self.latest_info_texts):
                color = (255, 255, 255)  # 默认白色
                
                if "UPDATED" in text:
                    color = (0, 255, 255)  # 黄色表示更新
                elif "No Detection" in text:
                    color = (0, 0, 255)  # 红色表示未检测到
                elif "Detected" in text:
                    color = (0, 255, 0)  # 绿色表示检测到
                elif "Conditions not met" in text:
                    color = (0, 165, 255)  # 橙色表示条件不满足
                elif "Detection Rate" in text or "Display" in text:
                    color = (255, 255, 0)  # 青色表示频率信息
                
                cv2.putText(display_image, text, (10, y_offset + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # 显示图像
            cv2.imshow("Rotated Box Depth Estimation", display_image)
            cv2.waitKey(1)
            
        except Exception as e:
            rospy.logerr(f"图像显示错误: {e}")

    def run(self):
        rospy.spin()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        node = RotatedBoxDepthEstimation()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"节点启动失败: {e}")