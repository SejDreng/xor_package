#!/usr/bin/env python3

from rclpy.node import Node
import rclpy

# message types
from std_msgs.msg import UInt8
from std_msgs.msg import UInt8MultiArray
from std_msgs.msg import Float32


class listener_node(Node):
    def __init__(self):
        super().__init__('listener_node')
        self.subscription = self.create_subscription(
            UInt8,
            'topic_result',
            self.read_topic_callback,
            10
        )
        self.subscription  # prevent unused variable warning
    
    def read_topic_callback(self, msg):
        print(f'Received message: {msg.data}')
        # repeat listen



def main():
    print('Hi from sub_node.')

    rclpy.init()
    node = listener_node()

    rclpy.spin(node)
    
    # If we ever exit the spin loop
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()