#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/Joy.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/CommandLong.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>
#include <mavros_msgs/PositionTarget.h>
#include <mavros_msgs/RCIn.h>
#include "quadrotor_msgs/PositionCommand.h"
#include <nav_msgs/Odometry.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#define VELOCITY2D_CONTROL 0b101111000111

bool allow_yaw = true;

class Ctrl
{
public:
    Ctrl();
    void state_cb(const mavros_msgs::State::ConstPtr &msg);
    void position_cb(const nav_msgs::Odometry::ConstPtr &msg);
    void target_cb(const geometry_msgs::PoseStamped::ConstPtr& msg);
    void twist_cb(const quadrotor_msgs::PositionCommand::ConstPtr& msg);
    void control(const ros::TimerEvent&);
    double calculateYawFromQuat(const geometry_msgs::Quaternion& q);
    
    ros::NodeHandle nh;
    visualization_msgs::Marker trackpoint;
    quadrotor_msgs::PositionCommand ego;
    tf::StampedTransform ts;
    tf::TransformBroadcaster tfBroadcasterPointer;
    unsigned short velocity_mask = VELOCITY2D_CONTROL;
    mavros_msgs::PositionTarget current_goal;
    mavros_msgs::RCIn rc;
    nav_msgs::Odometry position_msg;
    geometry_msgs::PoseStamped target_pos;
    mavros_msgs::State current_state;
    
    // 新增成员变量
    double target_yaw = 0.0;
    ros::Time last_twist_time;
    bool is_last_point = false;
    static constexpr double TIMEOUT_THRESHOLD = 0.5; // 0.5秒无新消息视为最终点

    float position_x, position_y, position_z, now_x, now_y, now_yaw, current_yaw, targetpos_x, targetpos_y;
    float ego_pos_x, ego_pos_y, ego_pos_z, ego_vel_x, ego_vel_y, ego_vel_z, ego_a_x, ego_a_y, ego_a_z, ego_yaw, ego_yaw_rate;
    bool receive, get_now_pos;
    bool have_odom = false;
    ros::Subscriber state_sub, twist_sub, target_sub, position_sub;
    ros::Publisher local_pos_pub, pubMarker;
    ros::Timer timer;
};

Ctrl::Ctrl() : last_twist_time(ros::Time(0))
{
    timer = nh.createTimer(ros::Duration(0.02), &Ctrl::control, this);
    state_sub = nh.subscribe("/mavros/state", 10, &Ctrl::state_cb, this);
    position_sub = nh.subscribe("/mavros/local_position/odom", 10, &Ctrl::position_cb, this);
    target_sub = nh.subscribe("move_base_simple/goal", 10, &Ctrl::target_cb, this);
    twist_sub = nh.subscribe("/position_cmd", 10, &Ctrl::twist_cb, this);
    local_pos_pub = nh.advertise<mavros_msgs::PositionTarget>("/mavros/setpoint_raw/local", 1);
    pubMarker = nh.advertise<visualization_msgs::Marker>("/track_drone_point", 5);
    get_now_pos = false;
    receive = false;
}

double Ctrl::calculateYawFromQuat(const geometry_msgs::Quaternion& q)
{
    tf2::Quaternion tf_q(q.x, q.y, q.z, q.w);
    tf2::Matrix3x3 m(tf_q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);
    return yaw;
}

void Ctrl::state_cb(const mavros_msgs::State::ConstPtr& msg)
{
    current_state = *msg;
}

void Ctrl::position_cb(const nav_msgs::Odometry::ConstPtr& msg)
{
    position_msg = *msg;
    tf2::Quaternion quat;
    tf2::convert(msg->pose.pose.orientation, quat);
    double roll, pitch, yaw;
    tf2::Matrix3x3(quat).getRPY(roll, pitch, yaw);
    
    ts.stamp_ = msg->header.stamp;
    ts.frame_id_ = "world";
    ts.child_frame_id_ = "drone_frame";
    ts.setRotation(tf::Quaternion(msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, 
                                 msg->pose.pose.orientation.z, msg->pose.pose.orientation.w));
    ts.setOrigin(tf::Vector3(msg->pose.pose.position.x, msg->pose.pose.position.y, 
                           msg->pose.pose.position.z));
    tfBroadcasterPointer.sendTransform(ts);
    
    if (!get_now_pos) 
    {
        now_x = position_msg.pose.pose.position.x;
        now_y = position_msg.pose.pose.position.y;
        tf2::convert(msg->pose.pose.orientation, quat);
        now_yaw = yaw;
        get_now_pos = true;
    }
    position_x = position_msg.pose.pose.position.x;
    position_y = position_msg.pose.pose.position.y;
    position_z = position_msg.pose.pose.position.z;
    current_yaw = yaw;
    have_odom = true;
}

void Ctrl::target_cb(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
    receive = true;
    target_pos = *msg;
    targetpos_x = target_pos.pose.position.x;
    targetpos_y = target_pos.pose.position.y;
    target_yaw = calculateYawFromQuat(target_pos.pose.orientation);
}

void Ctrl::twist_cb(const quadrotor_msgs::PositionCommand::ConstPtr& msg)
{
    ego = *msg;
    ego_pos_x = ego.position.x;
    ego_pos_y = ego.position.y;
    ego_pos_z = ego.position.z;
    ego_vel_x = ego.velocity.x;
    ego_vel_y = ego.velocity.y;
    ego_vel_z = ego.velocity.z;
    ego_yaw = ego.yaw;
    ego_yaw_rate = ego.yaw_dot;
    last_twist_time = ros::Time::now();
}

void Ctrl::control(const ros::TimerEvent&)
{
    if(!have_odom)
    {
        std::cout<<"---------------no odom!!-------------"<<std::endl;
        return;
    }

    // 检测最终点
    ros::Duration time_since_last = ros::Time::now() - last_twist_time;
    is_last_point = time_since_last.toSec() > TIMEOUT_THRESHOLD;

    if(!receive)
    {
        current_goal.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;
        current_goal.header.stamp = ros::Time::now();
        current_goal.type_mask = velocity_mask;
        current_goal.velocity.x = (now_x - position_x)*1;
        current_goal.velocity.y = (now_y - position_y)*1;
        current_goal.velocity.z = (1.5 - position_z)*1;
        current_goal.yaw = now_yaw;
        ROS_INFO("请等待\n");
    }

    if(receive)
    {
        current_goal.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;
        current_goal.header.stamp = ros::Time::now();
        current_goal.type_mask = velocity_mask;
        
        // 最终点处理逻辑
        if(is_last_point)
        {
            // 计算yaw差（考虑2π环绕）
            double yaw_diff = fmod(fabs(target_yaw - ego_yaw), 2*M_PI);
            if(yaw_diff > M_PI) yaw_diff = 2*M_PI - yaw_diff;
            
            if(yaw_diff > 0.087) // 约5度阈值
            {
                current_goal.yaw = target_yaw;
                // 停止平动，只保留高度控制
                current_goal.velocity.x = (0.8 * ego_vel_x + (ego_pos_x - position_x)*1);
                current_goal.velocity.y = (0.8 * ego_vel_y + (ego_pos_y - position_y)*1);
                current_goal.velocity.z = (ego_pos_z - position_z)*1;
            }
            else
            {
                current_goal.yaw = allow_yaw ? ego_yaw : now_yaw;
                current_goal.velocity.x = (0.8 * ego_vel_x + (ego_pos_x - position_x)*1);
                current_goal.velocity.y = (0.8 * ego_vel_y + (ego_pos_y - position_y)*1);
                current_goal.velocity.z = (ego_pos_z - position_z)*1;
            }
        }
        else
        {
            current_goal.velocity.x = (0.8 * ego_vel_x + (ego_pos_x - position_x)*1);
            current_goal.velocity.y = (0.8 * ego_vel_y + (ego_pos_y - position_y)*1);
            current_goal.velocity.z = (ego_pos_z - position_z)*1;
            current_goal.yaw = allow_yaw ? ego_yaw : now_yaw;
        }
        
        ROS_INFO("已触发控制器，当前EGO规划速度：vel_x = %.2f", 
               sqrt(pow(current_goal.velocity.x, 2)+pow(current_goal.velocity.y, 2)));
    }
    
    local_pos_pub.publish(current_goal);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "cxr_egoctrl_v1");
    setlocale(LC_ALL,"");
    Ctrl ctrl;
    ros::spin();
    return 0;
}
