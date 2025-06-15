#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import math

class CircleDepthEstimator:
    def __init__(self):
        self.bridge = CvBridge()
        
        # 加载YOLO模型
        self.model = YOLO('/home/amov/Downloads/yolo11n-lamp-10.0x-finetune.pt')
        
        # 相机内参
        self.K = np.array([913.9072265625, 0.0, 654.9307250976562, 
                           0.0, 914.03125, 365.23223876953125, 
                           0.0, 0.0, 1.0]).reshape(3, 3)
        
        # 采样参数
        self.sample_interval = 5  # 采样间隔（度）
        self.sample_count = int(180 / self.sample_interval)  # 采样次数
        self.k = 1.0  # 四分位距的k值
        
        # 迭代参数
        self.max_iterations = 4  # 最大迭代次数
        self.depth_diff_threshold = 0.1  # 深度差阈值 (m)
        self.depth_ends_diff_threshold = 0.25  # 线段两端深度差阈值 (m)
        self.max_bad_lines_ratio = 0.70  # 允许的最大不合格线段比例
        
        self.aspect_ratio_min = 0.90
        self.aspect_ratio_max = 1.10
        
        # 最新的彩色图像和深度图像
        self.color_image = None
        self.depth_image = None
        self.got_color = False
        self.got_depth = False
        
        # 处理状态
        self.status = "OK"
        self.status_details = ""
        
        # 订阅话题
        self.color_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.color_callback)
        self.depth_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_callback)
        
    
        
        # 处理定时器
        self.timer = rospy.Timer(rospy.Duration(0.0001), self.process_images)
    
    def color_callback(self, msg):
        self.color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.got_color = True
    
    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
        self.got_depth = True
    
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
    
    def process_images(self, event):
        if not (self.got_color and self.got_depth):
            return
        
        # 重置状态
        self.status = "OK"
        self.status_details = ""
        
        # 创建可视化图像
        vis_img = self.color_image.copy()
        
        # 使用YOLO模型进行检测
        results = self.model(self.color_image,conf=0.9)
        
        # 提取检测框
        boxes = results[0].boxes

        if len(boxes) == 0:
            self.status = "FAILED"
            self.status_details = "No circle detected"
            cv2.putText(vis_img, self.status_details, (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Detection Result', vis_img)
            cv2.waitKey(1)
            return
        
        # 如果有多个检测框，取最大的一个
        if len(boxes) > 1:
            # 计算每个框的面积
            areas = []
            for box in boxes.xyxy.cpu().numpy():
                area = (box[2] - box[0]) * (box[3] - box[1])
                areas.append(area)
            
            # 找到最大面积的框
            max_idx = np.argmax(areas)
            box = boxes.xyxy.cpu().numpy()[max_idx]
            name = results[0].names[max_idx]
        else:
            box = boxes.xyxy.cpu().numpy()[0]
            name = results[0].names[0]
        
        # 计算检测框的宽度和高度
        box_width = box[2] - box[0]
        box_height = box[3] - box[1]
        
        # 计算长宽比
        aspect_ratio = box_width / box_height if box_height > 0 else 0
        
        # 检查长宽比是否在允许范围内
        if aspect_ratio < self.aspect_ratio_min or aspect_ratio > self.aspect_ratio_max:
            self.status = "FAILED"
            self.status_details = f"Invalid aspect ratio: {aspect_ratio:.2f} (should be between {self.aspect_ratio_min} and {self.aspect_ratio_max})"
            cv2.putText(vis_img, self.status_details, (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(vis_img, name, (50, 250), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(vis_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
            cv2.imshow('Detection Result', vis_img)
            cv2.waitKey(1)
            return
        
        # 计算圆心坐标
        cx = int((box[0] + box[2]) / 2)
        cy = int((box[1] + box[3]) / 2)
        
        # 在图像上绘制圆心
        cv2.circle(vis_img, (cx, cy), 5, (0, 255, 0), -1)
        
        # 使用较小的边长作为初始线段长度
        box_size = min(box_width, box_height)
        initial_line_length = box_size
        
        # 计算迭代步长
        step_size = initial_line_length * 0.05
        
        # 迭代调整线段长度
        current_length = initial_line_length
        prev_depth = None
        valid_depth = None
        valid_sample_points = None
        valid_line_length = None
        valid_paired_samples = None
        valid_bad_lines_count = None
        valid_bad_lines_ratio = None
        
        # 记录迭代过程中的失败原因
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
        
        # 检查是否找到有效的深度
        if valid_depth is None:
            self.status = "FAILED"
            self.status_details = "Failed to find valid depth after iterations"
            
            # 显示迭代失败的具体原因

            for i, failure in enumerate(iteration_failures[-min(2, len(iteration_failures)):]):
                cv2.putText(vis_img, failure, (50, 90 + i*40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(vis_img, name, (50, 290), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.putText(vis_img, self.status_details, (50, 250), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(vis_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 165, 0), 2)
            cv2.imshow('Detection Result', vis_img)
            cv2.waitKey(1)
            return

        # 在图像上绘制采样点和线段
        for x, y, point_type in valid_sample_points:
            color = (255, 0, 0) if point_type == 'inner' else (0, 0, 255)
            cv2.circle(vis_img, (x, y), 3, color, -1)
        
        # 绘制线段
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
                cv2.line(vis_img, (inner_x, inner_y), (outer_x, outer_y), line_color, 1)
                
                # 如果是不合格线段，记录角度
                if is_bad_line:
                    angle = math.atan2(outer_y - inner_y, outer_x - inner_x) * 180 / math.pi
                    bad_line_angles.append(angle)
        
        # 在图像上显示结果
        cv2.putText(vis_img, f"STATUS: {self.status}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(vis_img, f"Depth: {valid_depth:.3f}m", (50, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(vis_img, f"Bad lines: {valid_bad_lines_count}/{len(valid_paired_samples)} ({valid_bad_lines_ratio*100:.1f}%)", 
                    (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(vis_img, f"Line length: {valid_line_length:.1f}px", (50, 170), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(vis_img, f"Aspect ratio: {aspect_ratio:.2f}", (50, 210), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(vis_img, name, (50, 250), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # 如果有不合格线段，显示它们的角度
        if bad_line_angles:
            bad_angles_text = "Bad line angles: " + ", ".join([f"{angle:.0f}°" for angle in bad_line_angles[:3]])
            if len(bad_line_angles) > 3:
                bad_angles_text += f" and {len(bad_line_angles) - 3} more"
            cv2.putText(vis_img, bad_angles_text, (50, 290), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # 绘制检测框
        cv2.rectangle(vis_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 165, 0), 2)
        
        # 显示结果
        cv2.imshow('Detection Result', vis_img)
        cv2.waitKey(1)

if __name__ == '__main__':
    rospy.init_node('circle_depth_estimator')
    estimator = CircleDepthEstimator()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()