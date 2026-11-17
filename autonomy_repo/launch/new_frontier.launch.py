#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os
import sys

def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    path_to_test = 'src/Section2/autonomy_repo/scripts'
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            IncludeLaunchDescription(
                PathJoinSubstitution(
                    [FindPackageShare("asl_tb3_sim"), "launch", "rviz.launch.py"]
                ),
                launch_arguments={
                    "config": PathJoinSubstitution(
                        [
                            FindPackageShare("autonomy_repo"),
                            "rviz",
                            "default.rviz",
                        ]
                    ),
                    "use_sim_time": use_sim_time,
                }.items(),
            ),
            TimerAction(
            period=5.0,
            actions=[
                Node(
                    executable=sys.executable,
                    arguments=[os.path.join(path_to_test, 'frontier_exploration.py')],
                    additional_env={'PYTHONUNBUFFERED': '1'},
                    name='FrontierExplore',
                )
            ]),
            # relay RVIZ goal pose to some other channel
            Node(
                executable="rviz_goal_relay.py",
                package="asl_tb3_lib",
                parameters=[
                    {"output_channel": "/cmd_nav"},
                ],
            ),
            # state publisher for turtlebot
            Node(
                executable="state_publisher.py",
                package="asl_tb3_lib",
            ),
            # student's heading controller node
            Node(
                executable="navigator.py",
                package="autonomy_repo", 
                parameters = [{"use_sim_time": use_sim_time},],
            ),
             Node(
                executable="lab7_explore.py",
                package="autonomy_repo", 
                parameters = [{"use_sim_time": use_sim_time},],
            ),
        ]
    )
