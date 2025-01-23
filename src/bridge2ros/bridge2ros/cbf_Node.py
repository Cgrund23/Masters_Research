#!/usr/bin/env python3
import rclpy
import math
import numpy as np
import sys
sys.path.append("/home/parallels/ackerman/src/bridge2ros/bridge2ros")
from dataclasses import dataclass
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray 
from sensor_msgs.msg import LaserScan
from CBF import CBF
from Visulise_kernals import MyFig



class Controller_Node(Node):
    def __init__(self):
        super().__init__('Controller_Node')
        self.subscription = self.create_subscription(
            Odometry,
            '/model/Car/odometry',
            self.pose_callback,
            10)
        
        
        self.subscription = self.create_subscription(
            LaserScan,
            'lidar',
            self.lidar_pose_callback,
            10)
        

        class params():
            dt: float = 1/10 # 10ms

            # Car info

            v: float = 1.0 # velocity
            u_max: float = [1,1.54] # max speed,angle
            u_min: float = [0.25,-1.54] # min speed,angle

            # Starting pose
            beta: float = 0
            gamma: float = 0
            theta: float = 0    
            
            x0: float = 0  # Start x
            y0: float = 0   # Start y
            
            # TODO get from model
            xdim: float = 4
            udim: float = 2
            lf: float = 1
            lr: float = 1

            # Obstacle position
            #TODO will be from lidar and continually updated figured out

            cbf_gamma: float = 1

            # Desired target point 
            #TODO this will still exist need to find a way to relate global to local

            weightslack:float = 10

            cbfrate:float = 1

        self.params = params
        self.CBFobj = CBF(params)
        # Publisher and Subscriber

        self.my_vel_command = self.create_publisher(Twist, "cmd_vel", 10)
        self.visual = self.create_publisher(Float64MultiArray, "visual", 10)

    def pose_callback(self,msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        theta = msg.pose.pose.orientation.x
        angle_rate = msg.twist.twist.angular.z
        print(theta)
        v = msg.twist.twist.linear.x
        self.CBFobj.update_pose(x,y,theta,v)


    def lidar_pose_callback(self, msg):
        r = np.array(msg.ranges)  # DistanceSS
        numpoints = len(r)
        self.params.ranges = r
        self.angle = (msg.angle_max - msg.angle_min)/numpoints
        angle = np.arange(msg.angle_min, msg.angle_max, self.angle)
        self.params.array = angle
        self.params.ranges = r
        self.CBFobj.setObjects(r,angle)
        
        try:
            x,h,dcbf = (self.CBFobj.constraints_cost())
            h = np.vstack((h,h))
            self.visulise(h) 
            self.send_vel(0.5,0.5)
        except:
            pass
        
    def visulise(self,B):
        msg = Float64MultiArray()
        A = self.params.array
        A[np.isinf(A)] = 12
        A = np.append(A,self.params.ranges)
        B = B.flatten()
        A = np.append(A,B)
        A = A.tolist()
        msg.data = A
        self.visual.publish(msg)


    def send_vel(self,x,z):
        my_msg = Twist()
        my_msg.linear.x = float(x)
        my_msg.angular.z = float(z)
        # self.get_logger().info('msg =: "%s"' % my_msg)
        self.my_vel_command.publish(my_msg)

def main(args=None):
    rclpy.init(args=args)
    controller = Controller_Node()
    controller.get_logger().info("Hello friend!")
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
