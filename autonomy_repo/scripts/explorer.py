#!/usr/bin/env python3

from asl_tb3_lib.navigation import BaseNavigator, TrajectoryPlan
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState
from asl_tb3_lib.grids import snap_to_grid, StochOccupancyGrid2D
from nav_msgs.msg import OccupancyGrid, Path
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
import typing as T

class Explorer(Node):
    def __init__(self):
        
        super().__init__("explorer_node")
        self.occupancy: T.Optional[StochOccupancyGrid2D] = None
        self.state: T.Optional[TurtleBotState] = None
        self.frontier: T.Optional[StochOccupancyGrid2D] = None

        self.cmd_nav_pub = self.create_publisher(TurtleBotState, "/cmd_nav", 10)
        self.map_sub = self.create_subscription(OccupancyGrid, "/map", self.map_callback, 10)
        self.state_sub = self.create_subscription(TurtleBotState, "/state", self.state_callback, 10)
        self.nav_success_sub = self.create_subscription(Bool, "/nav_success", self.success_callback, 10)
        self.timer_sub = self.create_timer(0.2, self.timer_callback)

    def timer_callback(self) -> None:
        self.timer_sub.cancel() # only needed upon startup
        self.frontier = self.explore(self.occupancy)
        start = np.array([self.state.x, self.state.y])
        goal, dist = self.closest_frontier(self.frontier, start)
        msg = TurtleBotState()
        msg.x, msg.y = goal[0], goal[1]
        msg.theta = 0
        self.cmd_nav_pub.publish(msg)
        
    def map_callback(self, msg: OccupancyGrid) -> None:
        """ Callback triggered when the map is updated

        Args:
            msg (OccupancyGrid): updated map message
        """
        self.occupancy = StochOccupancyGrid2D(
            resolution=msg.info.resolution,
            size_xy=np.array([msg.info.width, msg.info.height]),
            origin_xy=np.array([msg.info.origin.position.x, msg.info.origin.position.y]),
            window_size=9,
            probs=msg.data,
        )

    def state_callback(self, msg: TurtleBotState) -> None:
        """ callback triggered when receiving latest turtlebot state

        Args:
            msg (TurtleBotState): latest turtlebot state
        """
        self.state = msg
    
    def success_callback(self, msg: Bool) -> None:
        if msg == True:
            self.frontier = self.explore(self.occupancy)
        start = np.array([self.state.x, self.state.y])
        goal, dist = self.closest_frontier(self.frontier, start)
        msg = TurtleBotState()
        msg.x, msg.y = goal[0], goal[1]
        msg.theta = 0
        self.cmd_nav_pub.publish(msg)
    
    def explore(self, occupancy, window_size=11, min_unknown=0.25, min_free=0.35, occupied_thresh=50):
        """
        Frontier cells of an occupancy grid.
        A cell of occupancy.probs is unknown if its value is -1, occupied if it is >= occupied_thresh,
        and free otherwise. A cell is a frontier cell if, in the window_size x window_size window
        centered on it (cells outside the map are not counted, but the fractions are still of
        window_size^2),
            - at least min_unknown of the window_size^2 cells are unknown,
            - none of the cells are occupied, and
            - at least min_free of the window_size^2 cells are free.
        The cell itself may be free or unknown. Use the window_size argument, not occupancy.window_size.
        Output:
            frontier_states: (N, 2) array of (x, y) states of the frontier cells, in row-major order
            (increasing ky, then increasing kx, which is the order np.nonzero returns)

        HINT: work with grid indices and convert to states at the end with occupancy.grid2state.
            probs is indexed as probs[ky, kx], while grid2state expects (kx, ky).
        """
        probs = occupancy.probs
        ########## Code starts here ##########
        window_radius = int(window_size // 2);
        frontier_states = np.zeros((1, 2));
        for r in range(probs.shape[0]):
            for c in range(probs.shape[1]):
                
                # for each cell in map, check the window. tally # of unknowns and # of free cells. move on if we encounter an occupied cell
                unknowns = 0;
                frees = 0;
                occupied = False;
                
                for i in range(-window_radius, window_radius + 1):
                
                    if occupied == True:
                        break; # no need to check the rest of the window, move to next cell + window
                    if (r + i < 0) or (r + i > (probs.shape[0] - 1)):
                        continue; # don"t count cells too far up/down
                    
                    for j in range(-window_radius, window_radius + 1):
                        
                        if (c + j < 0) or (c + j > (probs.shape[1] - 1)):
                            continue; # don't count cells too far left/right
                        if probs[r + i][c + j] >= occupied_thresh:
                            # if just 1 cell in the window is occupied, then we want to move to the next cell
                            occupied = True;
                            break;
                        elif probs[r + i][c + j] == -1:
                            unknowns = unknowns + 1;
                        elif probs[r + i][c + j] < occupied_thresh:
                            frees = frees + 1;
                if unknowns >= min_unknown * window_size**2 and frees >= min_free * window_size**2 and occupied == False:
                    frontier_states = np.vstack((frontier_states, np.asarray([[int(c), int(r)]]))) 
        
        frontier_states = frontier_states.astype(int)              
        frontiers = np.zeros((1, 2));              
        for i in range(1, frontier_states.shape[0]): # the first entry is (0, 0)
            frontiers = np.vstack((frontiers, np.asarray(occupancy.grid2state(np.asarray([frontier_states[i][0], frontier_states[i][1]])))))                                            
        ########## Code ends here ##########
        return frontiers[1:][:]

    def closest_frontier(self, frontier_states, current_state):
        """
        Output:
            (state, distance): the frontier state closest to current_state in Euclidean distance,
            and that distance. If several states are equally close, return the first one.
            Returns (None, np.inf) if there are no frontier states.
        """
        ########## Code starts here ##########
        frontier_states = np.asarray(frontier_states)
        if frontier_states.shape[0] == 0:
            return (None, np.inf);

        min_dist = np.inf;
        min_idx = 0;
        for i in range(frontier_states.shape[0]):
            dist = np.linalg.norm(frontier_states[i][:] - current_state)
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        
        return (frontier_states[i][:], min_dist)

if __name__ == "__main__":
    rclpy.init()            # initialize ROS client library
    node = Explorer()       # create the node instance
    rclpy.spin(node)        # call ROS2 default scheduler
    rclpy.shutdown()        # clean up after node exits