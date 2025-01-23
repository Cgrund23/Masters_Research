#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.get_logger().info('HIP')
        self.subscription = self.create_subscription(
            LaserScan,
            'lidar',
            self.listener_callback,
            10)

        self.my_vel_command = self.create_publisher(Twist, "cmd_vel", 10)

    def listener_callback(self, msg):
        self.get_logger().info('min angle =: "%s"' % msg.angle_min)
        self.get_logger().info('max angle =: "%s"' % msg.angle_max)
        self.get_logger().info('msg =: "%s"' % msg.ranges)
        self.my_velocity_cont()

    def my_velocity_cont(self):
        my_msg = Twist()
        my_msg.linear.x = 10.
        my_msg.angular.z = .5
        self.get_logger().info('msg =: "%s"' % my_msg)
        self.my_vel_command.publish(my_msg)

def main(args=None):
    rclpy.init(args=args)
    node = MinimalSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
