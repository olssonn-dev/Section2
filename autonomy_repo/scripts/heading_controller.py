#!/usr/bin/env python3
import numpy as np
import rclpy
from asl_tb3_lib.control import BaseHeadingController
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState

class HeadingController(BaseHeadingController):
    
    def __init__(self):
        
        super().__init__("HeadingController")
        # self.kp = 2.0;
        self.declare_parameter("kp", 2.0)
        # don't need to establish our own publishers/subscribers, since those are inherited from BaseHeadingController
        
    @property
    def kp(self) -> float:
        return self.get_parameter("kp").value

    def compute_control_with_goal(self, state: TurtleBotState, goal: TurtleBotState) -> TurtleBotControl:
        
        error = wrap_angle(goal.theta - state.theta)
        omega = self.get_parameter("kp").value * error;
        msg = TurtleBotControl()
        msg.omega = omega;
        return msg;
        
if __name__ == "__main__":
    rclpy.init()        # initialize ROS2 context (must run before any other rclpy call)
    node = HeadingController()  # instantiate the headingController node
    rclpy.spin(node)    # Use ROS2 built-in scheduler for executing the node
    rclpy.shutdown()    # cleanly shutdown ROS2 context
