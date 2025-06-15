#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import torch
import math

class CircleDetector:
    def __init__(self):
        rospy.init_node('circle_detector', anonymous=True)
        
        # 初始化CvBridge
        self.bridge = CvBridge()
        
        # 加载YOLO模型
        self.model_path = '/home/amov/Downloads/yolo11n-lamp-10.0x-finetune.pt'
        self.model = YOLO(self.model_path)
        
        # 订阅ROS话题
        self.color_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.color_callback)
        self.depth_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_callback)
        

        
        # 初始化数据
        self.color_image = None
        self.depth_image = None
        self.color_timestamp = None
        self.depth_timestamp = None
        self.sync_threshold = 0.05  # 100ms同步阈值
        
        # 深度处理参数
        self.depth_invalid_threshold = 0.01  # 小于此值的深度被视为无效
        self.depth_diff_threshold = 0.1  # 深度差异阈值
        
        rospy.loginfo("Circle detector initialized with model: %s", self.model_path)
    
    def color_callback(self, msg):
        try:
            self.color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.color_timestamp = msg.header.stamp.to_sec()
            self.process_frames()
        except Exception as e:
            rospy.logerr(f"Error processing color image: {e}")
    
    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.depth_timestamp = msg.header.stamp.to_sec()
            self.process_frames()
        except Exception as e:
            rospy.logerr(f"Error processing depth image: {e}")
    
    def is_valid_depth(self, depth):
        """检查深度值是否有效"""
        return (not math.isnan(depth)) and (depth > self.depth_invalid_threshold)
    
    def get_depth_at_point(self, x, y):
        """获取深度图像中指定点的深度值，处理异常值"""
        if x < 0 or y < 0 or x >= self.depth_image.shape[1] or y >= self.depth_image.shape[0]:
            return None
        
        # 取点周围5x5区域
        kernel_size = 5
        half_size = kernel_size // 2
        
        x1, y1 = max(0, int(x-half_size)), max(0, int(y-half_size))
        x2, y2 = min(self.depth_image.shape[1], int(x+half_size+1)), min(self.depth_image.shape[0], int(y+half_size+1))
        
        try:
            depth_roi = self.depth_image[y1:y2, x1:x2].astype(np.float32)
            
            # 处理NaN
            depth_roi_no_nan = depth_roi[~np.isnan(depth_roi)]
            
            # 过滤掉无效深度值
            valid_depths = depth_roi_no_nan[depth_roi_no_nan > self.depth_invalid_threshold]
            
            if len(valid_depths) == 0:
                return None
                
            # 使用中值滤波减少异常值影响
            median_depth = np.median(valid_depths)
            return median_depth
            
        except Exception as e:
            rospy.logerr(f"Error getting depth at point ({x},{y}): {e}")
            return None
    
    def process_frames(self):
        """处理图像和深度帧"""
        # 检查是否收到了彩色和深度图像
        if self.color_image is None or self.depth_image is None:
            return
        
        # 检查时间戳是否同步
        if self.color_timestamp is None or self.depth_timestamp is None:
            return
        
        if abs(self.color_timestamp - self.depth_timestamp) > self.sync_threshold:
            return
        
        try:
            # 复制图像用于绘制
            display_img = self.color_image.copy()
            
            # 使用YOLO模型进行检测
            results = self.model(self.color_image)
            
            # 获取所有预测框
            detection_boxes = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf.cpu().numpy()[0]
                    detection_boxes.append((int(x1), int(y1), int(x2), int(y2), confidence))
            
            # 如果没有检测到任何框，跳过
            if not detection_boxes:
                cv2.putText(display_img, "No detection", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow('Detection Result', display_img)
                cv2.waitKey(1)
                return
            
            # 如果有多个框，取最大的那个
            max_box = max(detection_boxes, key=lambda box: (box[2]-box[0])*(box[3]-box[1]))
            x1, y1, x2, y2, conf = max_box
            
            # 计算框的宽和高
            width = x2 - x1
            height = y2 - y1
            
            # 检查长宽比是否符合要求
            aspect_ratio = width / height
            if aspect_ratio > 1.10 or aspect_ratio < 0.90:
                cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(display_img, f"Bad aspect ratio: {aspect_ratio:.2f}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow('Detection Result', display_img)
                cv2.waitKey(1)
                return
            
            # 绘制检测框
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 计算初始参考圆心
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            cv2.circle(display_img, (center_x, center_y), 5, (255, 0, 0), -1)
            
            # 计算步长
            step_size = max(1, int(width * 0.05))
            
            # 向上平移查找合适的圆心
            upward_depths = []
            upward_positions = []
            
            init_depth = self.get_depth_at_point(center_x, center_y)
            if init_depth is None:
                cv2.putText(display_img, "Invalid depth at center", (20, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow('Detection Result', display_img)
                cv2.waitKey(1)
                return
                
            upward_depths.append(init_depth)
            upward_positions.append((center_x, center_y))
            
            upward_stop = False
            for i in range(4):
                new_y = center_y - (i + 1) * step_size
                curr_depth = self.get_depth_at_point(center_x, new_y)
                
                # 处理无效深度
                if curr_depth is None:
                    continue
                    
                upward_depths.append(curr_depth)
                upward_positions.append((center_x, new_y))
                
                # 检查深度差异
                if len(upward_depths) >= 2:
                    prev_depth = upward_depths[-2]
                    depth_diff = abs(curr_depth - prev_depth)
                    
                    if depth_diff < self.depth_diff_threshold:
                        upward_stop = True
                        break
            
            # 向下平移查找合适的圆心
            downward_depths = []
            downward_positions = []
            
            downward_depths.append(init_depth)
            downward_positions.append((center_x, center_y))
            
            downward_stop = False
            for i in range(6):
                new_y = center_y + (i + 1) * step_size
                curr_depth = self.get_depth_at_point(center_x, new_y)
                
                # 处理无效深度
                if curr_depth is None:
                    continue
                    
                downward_depths.append(curr_depth)
                downward_positions.append((center_x, new_y))
                
                # 检查深度差异
                if len(downward_depths) >= 2:
                    prev_depth = downward_depths[-2]
                    depth_diff = abs(curr_depth - prev_depth)
                    
                    if depth_diff < self.depth_diff_threshold:
                        downward_stop = True
                        break
            
            # 如果向上和向下都没有停止，跳过这一帧
            if not (upward_stop or downward_stop):
                cv2.putText(display_img, "No stable depth found in vertical direction", 
                           (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow('Detection Result', display_img)
                cv2.waitKey(1)
                return
            
            # 获取最小深度值的位置
            refined_center = None
            if upward_stop and downward_stop:
                # 如果两个方向都停止了，找最小深度
                all_depths = upward_depths + downward_depths[1:]  # 跳过第一个，因为跟upward_depths[0]重复
                all_positions = upward_positions + downward_positions[1:]
                
                if all_depths:
                    min_depth_idx = np.argmin(np.array(all_depths))
                    refined_center = all_positions[min_depth_idx]
            elif upward_stop and upward_depths:
                min_depth_idx = np.argmin(np.array(upward_depths))
                refined_center = upward_positions[min_depth_idx]
            elif downward_stop and downward_depths:
                min_depth_idx = np.argmin(np.array(downward_depths))
                refined_center = downward_positions[min_depth_idx]
            
            if refined_center is None:
                cv2.putText(display_img, "Could not determine refined center", 
                           (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow('Detection Result', display_img)
                cv2.waitKey(1)
                return
                
            ref_center_x, ref_center_y = refined_center
            cv2.circle(display_img, (ref_center_x, ref_center_y), 5, (0, 255, 255), -1)
            
            # 计算左右线段的初始长度和端点
            half_width = width // 2
            left_endpoint = (ref_center_x - half_width, ref_center_y)
            right_endpoint = (ref_center_x + half_width, ref_center_y)
            
            # 获取左右端点的深度
            left_depth = self.get_depth_at_point(left_endpoint[0], left_endpoint[1])
            right_depth = self.get_depth_at_point(right_endpoint[0], right_endpoint[1])
            
            # 处理无效深度
            if left_depth is None or right_depth is None:
                cv2.putText(display_img, "Invalid depth at endpoints", 
                           (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow('Detection Result', display_img)
                cv2.waitKey(1)
                return
            
            # 左边线段迭代
            left_lengths = [half_width]
            left_depths = [left_depth]
            
            for i in range(4):
                new_length = half_width - (i + 1) * step_size
                if new_length <= 0:
                    break
                    
                new_left_endpoint = (ref_center_x - new_length, ref_center_y)
                curr_left_depth = self.get_depth_at_point(new_left_endpoint[0], new_left_endpoint[1])
                
                if curr_left_depth is None:
                    continue
                    
                left_lengths.append(new_length)
                left_depths.append(curr_left_depth)
                
                # 检查深度差异
                if len(left_depths) >= 2:
                    prev_left_depth = left_depths[-2]
                    depth_diff = abs(curr_left_depth - prev_left_depth)
                    
                    if depth_diff < self.depth_diff_threshold:
                        break
            
            # 右边线段迭代
            right_lengths = [half_width]
            right_depths = [right_depth]
            
            for i in range(4):
                new_length = half_width - (i + 1) * step_size
                if new_length <= 0:
                    break
                    
                new_right_endpoint = (ref_center_x + new_length, ref_center_y)
                curr_right_depth = self.get_depth_at_point(new_right_endpoint[0], new_right_endpoint[1])
                
                if curr_right_depth is None:
                    continue
                    
                right_lengths.append(new_length)
                right_depths.append(curr_right_depth)
                
                # 检查深度差异
                if len(right_depths) >= 2:
                    prev_right_depth = right_depths[-2]
                    depth_diff = abs(curr_right_depth - prev_right_depth)
                    
                    if depth_diff < self.depth_diff_threshold:
                        break
            
            # 获取最终线段长度
            if not left_lengths or not right_lengths:
                cv2.putText(display_img, "Could not determine segment lengths", 
                           (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow('Detection Result', display_img)
                cv2.waitKey(1)
                return
                
            final_left_length = left_lengths[-1]
            final_right_length = right_lengths[-1]
            
            # 计算实际圆心
            actual_center_x = ref_center_x - final_left_length + (final_left_length + final_right_length) // 2
            actual_center_y = ref_center_y
            
            # 在图像上标记最终的圆心
            cv2.circle(display_img, (actual_center_x, actual_center_y), 8, (0, 0, 255), -1)
            cv2.line(display_img, (ref_center_x - final_left_length, ref_center_y), 
                    (ref_center_x + final_right_length, ref_center_y), (0, 255, 0), 2)
            
            # 获取实际圆心的深度
            center_depth = self.get_depth_at_point(actual_center_x, actual_center_y)
            depth_text = f"{center_depth:.3f}m" if center_depth is not None else "Invalid depth"
            
            # 添加调试信息
            cv2.putText(display_img, f"Confidence: {conf:.2f}", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_img, f"Aspect ratio: {aspect_ratio:.2f}", (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_img, f"Left length: {final_left_length}", (20, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_img, f"Right length: {final_right_length}", (20, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_img, f"Center depth: {depth_text}", (20, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 显示结果
            cv2.imshow('Detection Result', display_img)
            cv2.waitKey(1)
            
        except Exception as e:
            rospy.logerr(f"Error in process_frames: {e}")
    
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        detector = CircleDetector()
        detector.run()
    except rospy.ROSInterruptException:
        pass