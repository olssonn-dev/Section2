#!/usr/bin/env python3

from asl_tb3_lib.navigation import BaseNavigator, TrajectoryPlan
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_lib.tf_utils import quaternion_to_yaw
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState
from asl_tb3_lib.grids import snap_to_grid, StochOccupancyGrid2D
import numpy as np
from numpy import linalg
import rclpy
from scipy.interpolate import splev, splrep
import typing as T

class Navigator(BaseNavigator):
   
    def __init__(self):
        # don't need to establish our own publishers/subscribers, since those are inherited from BaseNavigator
        super().__init__("navigator_node")
        self.kp = 2.0;
        self.V_PREV_THRES = 0.0001 
        self.kpx = 2.0;
        self.kpy = 2.0;
        self.kdx = 2.0;
        self.kdy = 2.0;
        
    def compute_heading_control(self, state: TurtleBotState, goal: TurtleBotState) -> TurtleBotControl:
        """ Compute only orientation target (used for NavMode.ALIGN and NavMode.Park)

        Returns:
            TurtleBotControl: control target
        """
        error = wrap_angle(goal.theta - state.theta)
        omega = self.kp * error;
        msg = TurtleBotControl()
        msg.omega = omega;
        return msg;
    
    def reset(self) -> None: # so that control methods can run and use these vars
        self.V_prev = 0.
        self.om_prev = 0.
        self.t_prev = 0.
    
    def compute_trajectory_tracking_control(self, state: TurtleBotState, plan: TrajectoryPlan, t: float) -> TurtleBotControl:            
        """ Compute control target using a trajectory tracking controller

        Args:
            state (TurtleBotState): current robot state
            plan (TrajectoryPlan): planned trajectory
            t (float): current timestep

        Returns:
            TurtleBotControl: control command
        """
    
        dt = t - self.t_prev
        x = state.x;
        y = state.y;
        th = state.theta;
        x_d = splev(t, plan.path_x_spline, der=0)
        y_d = splev(t, plan.path_y_spline, der=0)
        xd_d = splev(t, plan.path_x_spline, der=1)
        yd_d = splev(t, plan.path_y_spline, der=1)
        xdd_d = splev(t, plan.path_x_spline, der=2)
        ydd_d = splev(t, plan.path_y_spline, der=2)

        ########## Code starts here ##########
        # avoid singularity
        if abs(self.V_prev) < self.V_PREV_THRES:
            self.V_prev = self.V_PREV_THRES

        xd = self.V_prev*np.cos(th)
        yd = self.V_prev*np.sin(th)

        # compute virtual controls
        u = np.array([xdd_d + self.kpx*(x_d-x) + self.kdx*(xd_d-xd),
                        ydd_d + self.kpy*(y_d-y) + self.kdy*(yd_d-yd)])

        # compute real controls
        J = np.array([[np.cos(th), -self.V_prev*np.sin(th)],
                            [np.sin(th), self.V_prev*np.cos(th)]])
        a, om = linalg.solve(J, u)
        V = self.V_prev + a*dt
        ########## Code ends here ##########
        
        """
        # apply control limits
        V = np.clip(V, -self.V_max, self.V_max)
        om = np.clip(om, -self.om_max, self.om_max)
        """

        # save the commands that were applied and the time
        self.t_prev = t
        self.V_prev = V
        self.om_prev = om

        msg = TurtleBotControl()
        msg.v = V;
        msg.omega = om;
        return msg
    
    def compute_trajectory_plan(self,
        state: TurtleBotState,
        goal: TurtleBotState,
        occupancy: StochOccupancyGrid2D,
        resolution: float,
        horizon: float, v_desired=0.15, spline_alpha=0.1) -> T.Optional[TrajectoryPlan]:
        """ Compute a trajectory plan using A* and cubic spline fitting

        Args:
            state (TurtleBotState): state
            goal (TurtleBotState): goal
            occupancy (StochOccupancyGrid2D): occupancy
            resolution (float): resolution
            horizon (float): horizon

        Returns:
            T.Optional[TrajectoryPlan]:
        """
        x_init_x = state.x;
        x_init_y = state.y;
        x_goal_x = goal.x;
        x_goal_y = goal.y;
        
        astar = AStar((x_init_x - horizon, x_init_y - horizon), (x_init_x + horizon, x_init_y + horizon), (x_init_x, x_init_y), (x_goal_x, x_goal_y), occupancy, resolution)
        # astar.solve();
        # path = astar.path;
        if not astar.solve() or len(astar.path) < 4:
            return None
        
        self.reset()
        
        # smooth plan and get timesteps
        path = np.asarray(astar.path)
        dist = np.diff(path, axis=0)
        dt = np.linalg.norm(dist, axis=1) / v_desired
        ts = np.concatenate(([0.0], np.cumsum(dt)))
                        
        path_x_spline = splrep(ts, path[:, 0], k = 3, s = spline_alpha)
        path_y_spline = splrep(ts, path[:, 1], k = 3, s = spline_alpha)
    
        return TrajectoryPlan(
            path=path,
            path_x_spline=path_x_spline,
            path_y_spline=path_y_spline,
            duration=ts[-1],
        )
     
class AStar(object):
        """Represents a motion planning problem to be solved using A*"""

        def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, occupancy, resolution=1,
                    heuristic="l2", heuristic_weight=1.0):
            self.statespace_lo = statespace_lo         # state space lower bound (e.g., [-5, -5])
            self.statespace_hi = statespace_hi         # state space upper bound (e.g., [5, 5])
            self.occupancy = occupancy                 # occupancy grid (a DetOccupancyGrid2D object)
            self.resolution = resolution               # resolution of the discretization of state space (cell/m)
            self.heuristic_type = heuristic            # norm used by the heuristic: "l1", "l2", or "linf"
            self.heuristic_weight = heuristic_weight   # weight w >= 1 on the heuristic (w > 1 gives weighted A*)
            self.x_offset = x_init                     
            self.x_init = self.snap_to_grid(x_init)    # initial state
            self.x_goal = self.snap_to_grid(x_goal)    # goal state

            self.closed_set = set()    # the set containing the states that have been visited
            self.open_set = set()      # the set containing the states that are condidate for future expension

            self.est_cost_through = {}  # dictionary of the estimated cost from start to goal passing through state (often called f score)
            self.cost_to_arrive = {}    # dictionary of the cost-to-arrive at state from start (often called g score)
            self.came_from = {}         # dictionary keeping track of each state's parent to reconstruct the path

            self.open_set.add(self.x_init)
            self.cost_to_arrive[self.x_init] = 0
            self.est_cost_through[self.x_init] = self.heuristic(self.x_init)

            self.path = None      # the final path as a list of states

        def is_free(self, x):
            """
            Checks if a give state x is free, meaning it is inside the bounds of the map and
            is not inside any obstacle.
            Inputs:
                x: state tuple
            Output:
                Boolean True/False
            Hint: self.occupancy is a DetOccupancyGrid2D object, take a look at its methods for what might be
                useful here
            """
            x = np.asarray(x)
            if x[0] <= self.statespace_hi[0] and x[0] >= self.statespace_lo[0] and x[1] <= self.statespace_hi[1] and x[1] >= self.statespace_lo[1]:
                return self.occupancy.is_free(x);
            
            return False;

        def distance(self, x1, x2):
            """
            Computes the Euclidean distance between two states. This is the cost of
            traveling along the edge between two neighboring states, and it is used
            for the cost-to-arrive. It should NOT depend on self.heuristic_type.
            Inputs:
                x1: First state tuple
                x2: Second state tuple
            Output:
                Float Euclidean distance

            HINT: This should take one line. Tuples can be converted to numpy arrays using np.array().
            """

            return np.linalg.norm(np.subtract(np.array(x2), np.array(x1)), ord = 2)

        def heuristic(self, x):
            """
            Computes the weighted heuristic estimate of the cost-to-go from state x
            to self.x_goal, i.e. w * ||x - x_goal||, where w = self.heuristic_weight
            and the norm is selected by self.heuristic_type:
                "l1":   sum of absolute coordinate differences
                "l2":   Euclidean norm
                "linf": largest absolute coordinate difference
            Input:
                x: state tuple
            Output:
                Float heuristic value

            HINT: np.linalg.norm accepts an `ord` argument. Raise a ValueError for
                an unrecognized self.heuristic_type.
            """
            ctg = np.subtract(np.array(x), np.array(self.x_goal));
            ctg_norm = np.linalg.norm(ctg, ord = float(self.heuristic_type[1:]));
            return self.heuristic_weight * ctg_norm;

        def snap_to_grid(self, x):
            """ Returns the closest point on a discrete state grid
            Input:
                x: tuple state
            Output:
                A tuple that represents the closest point to x on the discrete state grid
            """
            x = np.asarray(x)
            return (
                self.resolution * round((x[0] - self.x_offset[0]) / self.resolution) + self.x_offset[0],
                self.resolution * round((x[1] - self.x_offset[1]) / self.resolution) + self.x_offset[1],
            )

        def get_neighbors(self, x):
            """
            Gets the FREE neighbor states of a given state x. Assumes a motion model
            where we can move up, down, left, right, or along the diagonals by an
            amount equal to self.resolution.
            Input:
                x: tuple state
            Ouput:
                List of neighbors that are free, as a list of TUPLES

            HINTS: Use self.is_free to check whether a given state is indeed free.
                Use self.snap_to_grid (see above) to ensure that the neighbors
                you compute are actually on the discrete grid, i.e., if you were
                to compute neighbors by adding/subtracting self.resolution from x,
                numerical errors could creep in over the course of many additions
                and cause grid point equality checks to fail. To remedy this, you
                should make sure that every neighbor is snapped to the grid as it
                is computed.
            """
            neighbors = []
            d=self.resolution;
            potential_neighbors = [(x[0], x[1]+d), (x[0]+d, x[1]+d), (x[0]+d, x[1]), (x[0]+d, x[1]-d), (x[0], x[1]-d), (x[0]-d, x[1]-d), (x[0]-d, x[1]), (x[0]-d, x[1]+d)];
            for i in potential_neighbors:
                if self.is_free(i):
                    neighbors.append(i);
            
            snapped_nb = [self.snap_to_grid(nb) for nb in neighbors]
            
            return snapped_nb

        def find_best_est_cost_through(self):
            """
            Gets the state in open_set that has the lowest est_cost_through
            Output: A tuple, the state found in open_set that has the lowest est_cost_through
            """
            return min(self.open_set, key=lambda x: self.est_cost_through[x])

        def reconstruct_path(self):
            """
            Use the came_from map to reconstruct a path from the initial location to
            the goal location
            Output:
                A list of tuples, which is a list of the states that go from start to goal
            """
            path = [self.x_goal]
            current = path[-1]
            while current != self.x_init:
                path.append(self.came_from[current])
                current = path[-1]
            return list(reversed(path))

        def plot_path(self, fig_num=0, show_init_label=True):
            """Plots the path found in self.path and the obstacles"""
            if not self.path:
                return

            self.occupancy.plot(fig_num)

            solution_path = np.asarray(self.path)
            plt.plot(solution_path[:,0],solution_path[:,1], color="green", linewidth=2, label="A* solution path", zorder=10)
            plt.scatter([self.x_init[0], self.x_goal[0]], [self.x_init[1], self.x_goal[1]], color="green", s=30, zorder=10)
            if show_init_label:
                plt.annotate(r"$x_{init}$", np.array(self.x_init) + np.array([.2, .2]), fontsize=16)
            plt.annotate(r"$x_{goal}$", np.array(self.x_goal) + np.array([.2, .2]), fontsize=16)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)

            plt.axis([0, self.occupancy.width, 0, self.occupancy.height])

        def plot_tree(self, point_size=15):
            plot_line_segments([(x, self.came_from[x]) for x in self.open_set if x != self.x_init], linewidth=1, color="blue", alpha=0.2)
            plot_line_segments([(x, self.came_from[x]) for x in self.closed_set if x != self.x_init], linewidth=1, color="blue", alpha=0.2)
            px = [x[0] for x in self.open_set | self.closed_set if x != self.x_init and x != self.x_goal]
            py = [x[1] for x in self.open_set | self.closed_set if x != self.x_init and x != self.x_goal]
            plt.scatter(px, py, color="blue", s=point_size, zorder=10, alpha=0.2)

        def solve(self):
            """
            Solves the planning problem using the A* search algorithm. It places
            the solution as a list of tuples (each representing a state) that go
            from self.x_init to self.x_goal inside the variable self.path
            Input:
                None
            Output:
                Boolean, True if a solution from x_init to x_goal was found

            HINTS:  We're representing the open and closed sets using python's built-in
                    set() class. This allows easily adding and removing items using
                    .add(item) and .remove(item) respectively, as well as checking for
                    set membership efficiently using the syntax "if item in set".

                    Use self.distance for edge costs (cost-to-arrive) and
                    self.heuristic for the estimated cost-to-go, as in Algorithm 1
                    of the handout.
            """
            
            while len(self.open_set) > 0: # while there are more states to analyze, keep going
                
                x_current = self.find_best_est_cost_through()
                
                if x_current == self.x_goal: # final loop iteration will be when the last neighbor we're checking is the goal state
                    self.path = self.reconstruct_path()
                    return True, self.path
                
                self.open_set.remove(x_current) # this state is no longer one we want to analyze again, must remove so that while loop can stop appropriately. note that this state was already in the open set from when it was first analyzed as a neighbor
                self.closed_set.add(x_current) # after the coming analysis, this state will be done-with
                
                for nb in self.get_neighbors(x_current):
                    
                    if nb in self.closed_set:
                        continue; # we've already looked at this state when it was an x_current, meaning it's behind us on the vague route
                    
                    cost_to_come = self.distance(self.x_init, x_current) + self.distance(x_current, nb);
                    
                    if nb not in self.open_set: # this state has never been considered as a neighbor yet
                        self.open_set.add(nb)
                    elif cost_to_come > self.cost_to_arrive[nb]: # needs to have been analyzed before
                        continue; # no need to consider this point when the route from x_init to nb through x_current is longer than the previously-recorded shortest path to nb
                    
                    # we've now confirmed that the route to nb through x_current is the best so far
                    self.came_from[nb] = x_current;
                    self.cost_to_arrive[nb] = cost_to_come;
                    self.est_cost_through[nb] = cost_to_come + self.heuristic(nb);
                    
            return False; # if we've made it here, we went through all the nodes w/o reaching the goal               
    
if __name__ == "__main__":
    rclpy.init()            # initialize ROS client library
    node = Navigator()    # create the node instance
    rclpy.spin(node)        # call ROS2 default scheduler
    rclpy.shutdown()        # clean up after node exits