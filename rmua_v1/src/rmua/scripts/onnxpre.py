#!/usr/bin/env python
import rospy
import cv2
import numpy as np
import onnxruntime as ort
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

# 类别映射关系
CLASS_NAMES = {
    0: 'class_name1',
    1: 'class_name2',
    2: 'class_name3'
}

class YOLO11ROS:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('yolov11_detector', anonymous=True)
        
        # 参数配置
        self.model_path = rospy.get_param('~model_path', '/home/amov/Downloads/yolo11n-lamp-10.0x-finetune_ir9.onnx')
        self.conf_thres = rospy.get_param('~conf_thres', 0.5)
        self.iou_thres = rospy.get_param('~iou_thres', 0.45)
        
        # 初始化CV桥
        self.bridge = CvBridge()
        
        # 初始化YOLO模型（会阻塞直到加载完成）
        self.init_model()
        
        # 模型加载完成后才创建订阅
        # 修改 YOLOv11ROS 类中的订阅部分
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, 
                                        self.image_callback, 
                                        queue_size=1,
                                        buff_size=2**24)  # 增加这两个参数
        

    def init_model(self):
        """初始化ONNX模型，会阻塞直到加载完成"""
        rospy.loginfo("Loading YOLOv11 model...")
        
        # 配置ONNX Runtime
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        
        # 获取输入尺寸
        model_inputs = self.session.get_inputs()
        self.input_shape = model_inputs[0].shape
        self.input_width = self.input_shape[2]
        self.input_height = self.input_shape[3]
        
        # 初始化颜色调色板
        self.color_palette = np.random.uniform(0, 255, size=(len(CLASS_NAMES), 3))
        
        # 初始化缩放参数
        self.ratio = None
        self.dw = None
        self.dh = None
        rospy.loginfo(f"Model loaded. Input size: {self.input_width}x{self.input_height}")

    def preprocess(self, image):
        """图像预处理"""
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img, self.ratio, (self.dw, self.dh) = self.letterbox(img, (self.input_width, self.input_height))
        image_data = np.array(img) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))  # HWC -> CHW
        return np.expand_dims(image_data, axis=0).astype(np.float32)

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        """保持宽高比的图像缩放"""
        shape = img.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        
        # 计算缩放比例
        r = min(new_shape[0]/shape[0], new_shape[1]/shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        
        # 添加padding
        dw = new_shape[1] - new_unpad[0]
        dh = new_shape[0] - new_unpad[1]
        dw, dh = dw/2, dh/2  # 均分padding
        
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        left = int(round(dw - 0.1))
        right = int(round(dw + 0.1))
        return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color), (r, r), (dw, dh)

    def postprocess(self, outputs, original_image):
        """后处理检测结果"""
        outputs = np.transpose(np.squeeze(outputs[0]))
        boxes, scores, class_ids = [], [], []
        
        for i in range(outputs.shape[0]):
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)
            
            if max_score >= self.conf_thres:
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
                
                # 坐标转换
                x = (x - self.dw) / self.ratio[0]
                y = (y - self.dh) / self.ratio[1]
                w = w / self.ratio[0]
                h = h / self.ratio[1]
                
                boxes.append([int(x - w/2), int(y - h/2), int(w), int(h)])
                scores.append(max_score)
                class_ids.append(class_id)
        
        # NMS处理
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_thres, self.iou_thres)
        
        # 绘制结果
        for i in indices:
            self.draw_detection(original_image, boxes[i], scores[i], class_ids[i])
        return original_image

    def draw_detection(self, img, box, score, class_id):
        """绘制检测框"""
        x, y, w, h = box
        color = self.color_palette[class_id]
        
        # 绘制边界框
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        
        # 绘制标签
        label = f"{CLASS_NAMES[class_id]}: {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x, y-20), (x+tw, y), color, -1)
        cv2.putText(img, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    def image_callback(self, msg):
        """图像回调处理"""
        try:
            # 转换ROS图像
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 预处理 → 推理 → 后处理
            input_data = self.preprocess(cv_image)
            outputs = self.session.run(None, {self.session.get_inputs()[0].name: input_data})
            result_image = self.postprocess(outputs, cv_image)
            
            # 显示结果
            cv2.imwrite("result.jpg", result_image)
            cv2.imshow("YOLOv11 Detection", result_image)
            print(result_image.shape)
            cv2.waitKey(1)
            
        except Exception as e:
            rospy.logerr(f"Processing error: {str(e)}")
    def run(self):
        """运行节点"""
        rospy.spin()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        detector = YOLO11ROS()
        detector.run()
    except rospy.ROSInterruptException:
        pass