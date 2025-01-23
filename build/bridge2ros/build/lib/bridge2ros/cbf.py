#!/usr/bin/env python3
import rclpy
import math
import numpy as np
import sys
sys.path.append("/home/parallels/ackerman/src/bridge2ros/bridge2ros")
from dataclasses import dataclass
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from CBF import cbf



class Controller_Node(Node):
    def __init__(self):
        super().__init__('Controller_Node')
        self.subscription = self.create_subscription(
            LaserScan,
            'lidar',
            self.pose_callback,
            10)
        self.subscription  # prevent unused variable warning

        class params(): 
            # Car info

            v: float = 0.5 # velocity
            u_max: float = math.pi # max yaw rate (left)
            u_min: float = -math.pi #min yaw rate (right)

            x0: float = 5.5  # Start x
            y0: float = 5.5   # Start y
            theta: float = 0    # Starting theta
            xdim: float = 1
            udim: float = 1

            # Obstacle position

            
            #xo: float = self.turtle2_pose.x
            #yo: float = self.turtle2_pose.y
            # Obstacle radius
            d: float = 1
            cbf_gamma0: float = .5
            # Desired target point
            #xd: float = desired_x 
            #yd: float = desired_y

            clfrate: float = 1
            weightslack:float = 10

            cbfrate:float = 1

        
        self .params = params
        # Publisher and Subscriber
        self.subscription = self.create_subscription(LaserScan,'lidar',self.pose_callback,10)
        self.my_vel_command = self.create_publisher(Twist, "/cmd_vel", 10)

    # Distance to points
    def pose_callback(self, msg):
        maxDist = 12
        r = np.array(msg.ranges)  # DistanceSS
        r[np.isinf(r)] = maxDist
        numpoints = len(r)
        self.angle = (msg.angle_max - msg.angle_min)/numpoints
        array = np.arange(0, numpoints*self.angle, self.angle)
        self.params.array = array
        self.params.ranges = r
        self.CBFobj = cbf(self.params)

def main(args=None):
    rclpy.init(args=args)
    controller = Controller_Node()
    controller.get_logger().info("Hello friend!")
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
