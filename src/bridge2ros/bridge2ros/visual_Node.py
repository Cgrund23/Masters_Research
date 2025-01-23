#!/usr/bin/env python3
import rclpy
import math
import numpy as np
import sys
sys.path.append("/home/parallels/ackerman/src/bridge2ros/bridge2ros")
from dataclasses import dataclass
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray 
from sensor_msgs.msg import LaserScan
from CBF import CBF
from Visulise_kernals import MyFig



class Visual_Node(Node):
    def __init__(self):
        super().__init__('Controller_Node')
        self.plot = MyFig()
        self.subscription = self.create_subscription(Float64MultiArray,'visual',self.graph,10)
    
    def graph(self,msg):
        data = msg.data
        array = np.array(data)
        angle = array[0:359]
        range = array[360:719]
        matrix = array[720:]
       
        length = int((matrix.shape[0])/2)
        cbf = matrix[0:(length)]
        dcbf = matrix[length+1:]
        self.plot.updateCBF(cbf)
        #self.plot.updateDCBF(dcbf)
        self.plot.updateLidar(angle,range)
        self.plot.show()
        

def main(args=None):
    rclpy.init(args=args)
    controller = Visual_Node()
    controller.get_logger().info("Pretty!")
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()