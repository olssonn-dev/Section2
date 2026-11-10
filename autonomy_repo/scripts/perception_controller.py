#!/usr/bin/env python3
import numpy as np
import rclpy
from asl_tb3_lib.control import BaseHeadingController
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState
from std_msgs.msg import Bool

class PerceptionController(BaseHeadingController):
    
    def __init__(self):
        
        super().__init__("PerceptionController")
        # self.kp = 2.0;
        self.declare_parameter("kp", 2.0)
        self.declare_parameter("active", True)
        self.image_detected = False
        self.image_detected_subscriber = self.create_subscription(Bool, "/detector_bool", self.image_detected_callback, 10)
        # don't need to establish our own publishers/subscribers, since those are inherited from BaseHeadingController
    
    def image_detected_callback(self, msg: Bool):
        self.image_detected = msg.data

    @property
    def kp(self) -> float:
        return self.get_parameter("kp").value
    
    @property
    def active(self) -> bool:
        return self.get_parameter("active").value

    def compute_control_with_goal(self, state: TurtleBotState, goal: TurtleBotState) -> TurtleBotControl:
        
        if self.image_detected:
            #error = wrap_angle(goal.theta - state.theta)
            #omega = self.get_parameter("kp").value * error
            omega = 0.0
        else:
            omega = 0.2
        msg = TurtleBotControl()
        msg.omega = omega
        return msg
        
if __name__ == "__main__":
    rclpy.init()        # initialize ROS2 context (must run before any other rclpy call)
    node = PerceptionController()  # instantiate the headingController node
    rclpy.spin(node)    # Use ROS2 built-in scheduler for executing the node
    rclpy.shutdown()    # cleanly shutdown ROS2 context
