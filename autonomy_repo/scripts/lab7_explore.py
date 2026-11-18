#!/usr/bin/env python3
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_lib.tf_utils import quaternion_to_yaw
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState
from asl_tb3_lib.grids import snap_to_grid, StochOccupancyGrid2D
import numpy as np
from numpy import linalg
import rclpy
from rclpy.node import Node
import typing as T
from std_msgs.msg import Bool
from nav_msgs.msg import OccupancyGrid, Path
from scipy.signal import convolve2d
import time

class FrontierExplore(Node):
   
    def __init__(self):
        # don't need to establish our own publishers/subscribers, since those are inherited from BaseNavigator
        super().__init__("frontier_explorer")
        self.nav_success_sub = self.create_subscription(Bool,"/nav_success",self.success_callback,10)
        self.state_sub = self.create_subscription(TurtleBotState,"/state",self.state_callback,10)
        self.map_sub = self.create_subscription(OccupancyGrid,"/map",self.mapping_callback,10)
        self.image_sub = self.create_subscription(Bool,"/detector_bool",self.image_detected_callback,10)
        self.newpose_pub = self.create_publisher(TurtleBotState,"/cmd_nav",10)
        self.occupancy = None
        self.frontier_states = []
        self.new_state = np.zeros((2,))
        self.index = None
        self.first_try = True
        self.stop_sign = False
        self.stop_detected = 0
        self.time_stopped = 0
        self.allow_new_stop_sign = True
        self.new_state_final = TurtleBotState()

    def image_detected_callback(self, msg: Bool):
        #self.get_logger().info(f'Value: {self.stop_sign}')
        if msg.data and self.stop_detected == 0 and self.allow_new_stop_sign:
            self.hb_timer = self.create_timer(5, self.hb_callback)
            self.stop_detected = self.stop_detected + 1 # only stops right when stop-sign is detected
            self.stop_sign = True
            self.allow_new_stop_sign = False
            self.newpose_pub.publish(self.current_state)

    def hb_callback(self):
        self.get_logger().info(f'Reached hb callback')
        self.stop_detected = 0
        self.stop_sign = False
        self.hb_timer.cancel()
        self.new_timer = self.create_timer(3,self.new_timer_callback)
        self.newpose_pub.publish(self.new_state_final)

    def new_timer_callback(self):
        self.get_logger().info(f'Reached timer callback')
        self.allow_new_stop_sign = True
        self.new_timer.cancel()

    def state_callback(self, msg: TurtleBotState):
        self.current_state = msg
        if self.first_try and self.occupancy is not None:
            self.newpose_pub.publish(self.current_state)
            self.frontier_states = self.explore(self.occupancy)
            current_state = np.array([self.current_state.x,self.current_state.y])
            self.new_state,distance,self.index = self.closest_frontier(self.frontier_states,current_state)
            new_state_final = TurtleBotState()
            new_state_final.theta = self.current_state.theta
            new_state_final.x = self.new_state[0]
            new_state_final.y = self.new_state[1]
            self.newpose_pub.publish(new_state_final)
            self.first_try = False


    def mapping_callback(self, msg: OccupancyGrid) -> None:

        self.occupancy = StochOccupancyGrid2D(
            resolution=msg.info.resolution,
            size_xy=np.array([msg.info.width, msg.info.height]),
            origin_xy=np.array([msg.info.origin.position.x, msg.info.origin.position.y]),
            window_size=9,
            probs=msg.data,
        )

    def success_callback(self, msg: Bool):
        if msg.data and self.stop_sign == False:
            "reached nav success"
            if self.occupancy is None:
                print("self occupancy is none")
                return
            self.frontier_states = self.explore(self.occupancy)
            current_state = np.array([self.current_state.x,self.current_state.y])
            self.new_state,distance,self.index = self.closest_frontier(self.frontier_states,current_state)
            self.new_state_final = TurtleBotState()
            self.new_state_final.theta = self.current_state.theta
            self.new_state_final.x = self.new_state[0]
            self.new_state_final.y = self.new_state[1]
            self.newpose_pub.publish(self.new_state_final)
        elif (msg.data == False) and (self.stop_sign == False):
            self.frontier_states = np.delete(self.frontier_states,self.index,axis=0)
            current_state = np.array([self.current_state.x,self.current_state.y])
            self.new_state,distance,self.index = self.closest_frontier(self.frontier_states,current_state)
            self.new_state_final = TurtleBotState()
            self.new_state_final.theta = self.current_state.theta
            self.new_state_final.x = self.new_state[0]
            self.new_state_final.y = self.new_state[1]
            self.newpose_pub.publish(self.new_state_final)

    def window_counts(self,mask, window_size):

        ########## Code starts here ##########
        window = np.ones((window_size, window_size))
        counts = convolve2d(mask, window, mode="same")
        ########## Code ends here ##########
        return counts

    def explore(self,occupancy, window_size=5, min_unknown=0.35, min_free=0.2, occupied_thresh=50):

        probs = occupancy.probs
        ########## Code starts here ##########
        occupied = probs >= occupied_thresh
        unknown = probs == -1
        free = (probs < occupied_thresh) & (probs > -1)

        occupied_counts = self.window_counts(occupied, window_size)
        free_counts = self.window_counts(free, window_size)
        unknown_counts = self.window_counts(unknown, window_size)

        frontier_states = []

        for i in range(probs.shape[0]):
            for j in range(probs.shape[1]):
                if (
                    unknown_counts[i, j] >= (min_unknown * (window_size**2))
                    and free_counts[i, j] >= (min_free * (window_size**2))
                    and occupied_counts[i, j] == 0 
                ):
                    new_state = occupancy.grid2state(np.array([i, j]))
                    new_state = new_state[::-1]
                    if occupancy.is_free(new_state):
                        frontier_states.append(new_state)
        ########## Code ends here ##########
        return np.array(frontier_states)

    def closest_frontier(self,frontier_states, current_state):

        ########## Code starts here ##########
        distance = np.inf
        state = None
        index = None
        print("Frontier states:", frontier_states.shape)

        for i in range(frontier_states.shape[0]):
            if np.linalg.norm(frontier_states[i, :] - current_state) < distance and np.linalg.norm(frontier_states[i, :] - current_state) > 1:
                state = frontier_states[i, :]
                distance = np.linalg.norm(frontier_states[i, :] - current_state)
                index = i
        return state, distance,index
        ########## Code ends here #########

               
if __name__ == "__main__":
    rclpy.init()            # initialize ROS client library
    node = FrontierExplore()    # create the node instance
    rclpy.spin(node)        # call ROS2 default scheduler
    rclpy.shutdown()        # clean up after node exits