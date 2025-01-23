#!/usr/bin/env python3
import rclpy
import numpy as np
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.get_logger().info('HI')
        self.subscription = self.create_subscription(
            LaserScan,
            'lidar',
            self.listener_callback,
            10)
        
        self.my_vel_command = self.create_publisher(Twist, "cmd_vel", 10)

    def listener_callback(self, msg):
        #self.get_logger().info('min angle =: "%s"' % msg.angle_min)
        #self.get_logger().info('max angle =: "%s"' % msg.angle_max)
        #self.get_logger().info('msg =: "%s"' % msg.ranges)
        self.ranges = np.array(msg.ranges)
        self.min_dist = msg.range_min

        self.send_vel()

    def send_vel(self):
        min_idx = np.unravel_index(np.argmin(self.ranges, axis=None), self.ranges.shape)
        self.get_logger().info('msg =: "%s"' % min_idx)
        my_msg = Twist()
        my_msg.linear.x = 5.0
        if ((min_idx[0]) > 180):
            my_msg.angular.z = -.5
        else:
            my_msg.angular.z = .5
        # self.get_logger().info('msg =: "%s"' % my_msg)
        self.my_vel_command.publish(my_msg)

def main(args=None):
    rclpy.init(args=args)
    node = MinimalSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
