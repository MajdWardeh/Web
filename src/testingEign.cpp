#include <iostream>
#include <ros/ros.h>
#include <gazebo_msgs/LinkStates.h>
#include <geometry_msgs/Pose.h>
#include <Eigen/Dense>
#include <nav_msgs/Odometry.h>
#include <eigen_conversions/eigen_msg.h>

Eigen::Affine3d current_pose_;
Eigen::Vector3d current_velocity_;

void uavOdomCallback(const nav_msgs::Odometry::ConstPtr& odom) {
  // store current position in our planner
  tf::poseMsgToEigen(odom->pose.pose, current_pose_);
  std::cout << current_pose_.rotation()<<std::endl;
  
  Eigen::Vector3d ea = current_pose_.rotation().eulerAngles(0, 1, 2); 
  std::cout << "to Euler angles:" << std::endl;
  std::cout << ea << std::endl;

  // store current vleocity
  tf::vectorMsgToEigen(odom->twist.twist.linear, current_velocity_);
}

int main(int argc, char **argv)
{

  ros::init(argc, argv, "tester");
  ros::NodeHandle n;
  ros::Subscriber sub_odom_ = n.subscribe("uav_pose", 1, uavOdomCallback);
  ros::spin();
  return 0;
}



