#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import math
import tf
from tf import TransformBroadcaster
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseStamped, Twist, Quaternion
from sensor_msgs.msg import Joy
from mavros_msgs.msg import State, PositionTarget, RCIn
from mavros_msgs.srv import CommandBool, CommandLong, SetMode
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import PositionCommand


VELOCITY2D_CONTROL = 0b001111000111
TIMEOUT_THRESHOLD = 0.5  # 0.5秒无新消息视为最终点

class Ctrl:
    def __init__(self):
        self.allow_yaw = True
        
        # 初始化ROS节点
        rospy.init_node('cxr_egoctrl_v1', anonymous=True)

        self.point1 = rospy.get_param('~point1', [21,0.0,3.8,1.57])
        
        
        # 初始化变量
        self.current_flag = 0
        self.trackpoint = Marker()
        self.ego = PositionCommand()
        self.current_goal = PositionTarget()
        self.rc = RCIn()
        self.position_msg = Odometry()
        self.target_pos = PoseStamped()
        self.current_state = State()
        
        self.target_yaw = 0.0
        self.last_twist_time = rospy.Time(0)
        self.is_last_point = False
        self.is_position_control = False
        self.EGO_MODE = False
        self.POSCTL_MODE = False
        self.OFFBOARD_MODE = False

        self.waiting_for_twist = False
        self.position_x = 0.0
        self.position_y = 0.0
        self.position_z = 0.0
        self.now_x = 0.0
        self.now_y = 0.0
        self.now_yaw = 0.0
        self.current_yaw = 0.0
        self.targetpos_x = 0.0
        self.targetpos_y = 0.0
        
        self.ego_pos_x = 0.0
        self.ego_pos_y = 0.0
        self.ego_pos_z = 0.0
        self.ego_vel_x = 0.0
        self.ego_vel_y = 0.0
        self.ego_vel_z = 0.0
        self.ego_yaw = 0.0
        self.ego_yaw_rate = 0.0
        
        self.receive = False
        self.get_now_pos = False
        self.have_odom = False
        self.again_ego_mode = False
        self.prev_ego_mode = False
        self.is_landing = False
        

        self.tfBroadcasterPointer = TransformBroadcaster()
        
        
        self.state_sub = rospy.Subscriber("/mavros/state", State, self.state_cb)
        self.position_sub = rospy.Subscriber("/mavros/local_position/odom", Odometry, self.position_cb)
        self.target_sub = rospy.Subscriber("move_base_simple/goal", PoseStamped, self.target_cb)
        self.twist_sub = rospy.Subscriber("/position_cmd", PositionCommand, self.twist_cb)
        

        self.local_pos_pub = rospy.Publisher("/mavros/setpoint_raw/local", PositionTarget, queue_size=1)
        self.pubMarker = rospy.Publisher("/track_drone_point", Marker, queue_size=5)

        self.set_mode_client = rospy.ServiceProxy('/mavros/set_mode', SetMode)

        self.timer = rospy.Timer(rospy.Duration(0.02), self.control)
        rospy.set_param('/IS_LANDING', False)
        rospy.set_param('/POSCTL_MODE', False)
        rospy.set_param('/CURRENT_FLAG', 0)
        
    def state_cb(self, msg):
        self.current_state = msg
        
    def position_cb(self, msg):
        self.position_msg = msg
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        
        # 发布TF变换
        self.tfBroadcasterPointer.sendTransform(
            (msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z),
            (orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w),
            rospy.Time.now(),
            "drone_frame",
            "world"
        )
        
        if not self.get_now_pos:
            self.now_x = msg.pose.pose.position.x
            self.now_y = msg.pose.pose.position.y
            self.now_yaw = yaw
            self.get_now_pos = True
            
        self.position_x = msg.pose.pose.position.x
        self.position_y = msg.pose.pose.position.y
        self.position_z = msg.pose.pose.position.z
        self.current_yaw = yaw
        self.have_odom = True
        
    def target_cb(self, msg):
        self.receive = True
        self.EGO_MODE = True
        self.is_position_control = False
        self.POSCTL_MODE = False
        rospy.set_param('/POSCTL_MODE', False)
        rospy.set_param('/EGO_MODE', self.EGO_MODE)
        rospy.loginfo(f"{self.EGO_MODE}")
        self.target_pos = msg
        self.targetpos_x = msg.pose.position.x
        self.targetpos_y = msg.pose.position.y
        orientation_q = msg.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (_, _, self.target_yaw) = euler_from_quaternion(orientation_list)
        
    def twist_cb(self, msg):
        self.ego = msg
        self.waiting_for_twist = False
        self.ego_pos_x = msg.position.x
        self.ego_pos_y = msg.position.y
        self.ego_pos_z = msg.position.z
        self.ego_vel_x = msg.velocity.x
        self.ego_vel_y = msg.velocity.y
        self.ego_vel_z = msg.velocity.z
        self.ego_yaw = msg.yaw
        self.ego_yaw_rate = msg.yaw_dot
        self.last_twist_time = rospy.Time.now()
        
    def control(self, event):
        if not self.have_odom:
            rospy.logwarn("---------------no odom!!-------------")
            return
            
        # 检测最终点
        self.current_flag = rospy.get_param('/CURRENT_FLAG', 0)
        self.OFFBOARD_MODE = rospy.get_param('/OFFBOARD_MODE', self.OFFBOARD_MODE)
        self.POSCTL_MODE = rospy.get_param('/POSCTL_MODE', self.POSCTL_MODE)
        self.EGO_MODE = rospy.get_param('/EGO_MODE', self.EGO_MODE)
        self.is_landing = rospy.get_param('/IS_LANDING', False)
        time_since_last = rospy.Time.now() - self.last_twist_time
        self.is_last_point = time_since_last.to_sec() > TIMEOUT_THRESHOLD
        current_ego_state = self.EGO_MODE
        mode_changed = (self.prev_ego_mode != current_ego_state)
        rospy.loginfo(f"{self.current_flag},{self.EGO_MODE},{self.POSCTL_MODE},{self.is_position_control},{self.is_landing}")
        """
        if self.current_flag == 5 and not self.is_position_control:
            # 获取当前速度信息
            current_vel = self.position_msg.twist.twist.linear
            current_angular = self.position_msg.twist.twist.angular
            
            
            # 计算速度绝对值
            vel_x_abs = abs(current_vel.x)
            vel_y_abs = abs(current_vel.y)
            vel_z_abs = abs(current_vel.z)
            angular_z_abs = abs(current_angular.z)
            
            # 计算yaw角度差
            target_yaw = self.point1[3]
            yaw_diff = abs(self.current_yaw - target_yaw)
            yaw_diff = min(yaw_diff, 2*math.pi - yaw_diff)
            rospy.loginfo(f"yaw_diff = {yaw_diff},{math.radians(5)}")
            
            # 检查所有条件
            if (vel_x_abs < 0.1 and vel_y_abs < 0.1 and vel_z_abs < 0.1 and
                angular_z_abs < 0.1 and yaw_diff < math.radians(5)):
                self.POSCTL_MODE = True
                rospy.set_param('/POSCTL_MODE', self.POSCTL_MODE)
                self.EGO_MODE = False
                rospy.loginfo("切换至POSCTL模式：速度和角度条件满足")
        """
        
        if not self.receive and self.is_landing is False:
            self.current_goal.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
            self.current_goal.header.stamp = rospy.Time.now()
            self.current_goal.type_mask = VELOCITY2D_CONTROL
            self.current_goal.velocity.x = (self.now_x - self.position_x) * 1
            self.current_goal.velocity.y = (self.now_y - self.position_y) * 1
            self.current_goal.velocity.z = (1.0 - self.position_z) * 1
            self.current_goal.yaw = self.now_yaw
            rospy.loginfo("请等待")
            self.local_pos_pub.publish(self.current_goal)

        elif self.EGO_MODE == True:

            if mode_changed:
                if not self.again_ego_mode:
                    self.waiting_for_twist = True
                    self.again_ego_mode = True
                    rospy.loginfo("进入等待指令状态")
                
            if self.waiting_for_twist:
                # 悬停控制逻辑
                self.current_goal.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
                self.current_goal.header.stamp = rospy.Time.now()
                self.current_goal.type_mask = VELOCITY2D_CONTROL
                self.current_goal.velocity.x = 0.0
                self.current_goal.velocity.y = 0.0
                self.current_goal.velocity.z = 0.0
                self.current_goal.yaw = self.current_yaw
                rospy.loginfo("悬停等待新指令")
                self.local_pos_pub.publish(self.current_goal)
                
            else:
                self.current_goal.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
                self.current_goal.header.stamp = rospy.Time.now()
                self.current_goal.type_mask = VELOCITY2D_CONTROL
                self.prev_ego_mode = current_ego_state
                
                if self.is_last_point:
                    # 计算yaw差
                    yaw_diff = abs(self.target_yaw - self.ego_yaw) % (2 * math.pi)
                    if yaw_diff > math.pi:
                        yaw_diff = 2 * math.pi - yaw_diff
                    
                    if yaw_diff > 0.087:  # 约5度阈值
                        self.current_goal.yaw = self.target_yaw
                        self.current_goal.yaw_rate = 0.15
                        self.current_goal.velocity.x = 0.8 * self.ego_vel_x + (self.ego_pos_x - self.position_x) * 1
                        self.current_goal.velocity.y = 0.8 * self.ego_vel_y + (self.ego_pos_y - self.position_y) * 1
                        self.current_goal.velocity.z = (self.ego_pos_z - self.position_z) * 1
                        self.local_pos_pub.publish(self.current_goal)
                        
                    else:
                        self.current_goal.yaw = self.ego_yaw if self.allow_yaw else self.now_yaw
                        self.current_goal.yaw_rate = 0.15
                        self.current_goal.velocity.x = 0.8 * self.ego_vel_x + (self.ego_pos_x - self.position_x) * 1
                        self.current_goal.velocity.y = 0.8 * self.ego_vel_y + (self.ego_pos_y - self.position_y) * 1
                        self.current_goal.velocity.z = (self.ego_pos_z - self.position_z) * 1
                        self.local_pos_pub.publish(self.current_goal)

                else:
                    self.current_goal.velocity.x = 0.8 * self.ego_vel_x + (self.ego_pos_x - self.position_x) * 1
                    self.current_goal.velocity.y = 0.8 * self.ego_vel_y + (self.ego_pos_y - self.position_y) * 1
                    self.current_goal.velocity.z = (self.ego_pos_z - self.position_z) * 1
                    self.current_goal.yaw = self.ego_yaw if self.allow_yaw else self.now_yaw
                    self.local_pos_pub.publish(self.current_goal)
                    
                
                rospy.loginfo("已触发控制器，当前EGO规划速度：vel_x = %.2f", 
                            math.sqrt(self.current_goal.velocity.x**2 + self.current_goal.velocity.y**2))
            
        elif (self.is_landing == True and not self.is_position_control and self.OFFBOARD_MODE == False):
            if self.current_state.mode != "POSCTL":
                resp = self.set_mode_client(0, "POSCTL")
                if resp.mode_sent:
                    rospy.loginfo("Position mode activated")
                    self.EGO_MODE = False
                    rospy.set_param("/EGO_MODE", self.EGO_MODE)
                    rospy.set_param("/POSCTL_MODE",True)
                    self.is_position_control = True
                else:
                    rospy.logwarn("Position mode switch rejected")
            
        
        elif self.OFFBOARD_MODE == True:
            rospy.loginfo("OFFBOARD MODE")
            return

        if not self.EGO_MODE:
            self.waiting_for_twist = False

        
        

if __name__ == '__main__':
    try:
        ctrl = Ctrl()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
