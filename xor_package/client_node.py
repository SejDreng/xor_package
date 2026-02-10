#!/usr/bin/env python3

from rclpy.node import Node
import rclpy

import random

# message types
from std_msgs.msg import UInt8
from std_msgs.msg import UInt8MultiArray
from std_msgs.msg import Float32

# service types
from xor_package.srv import XorRequest

class ml_client(Node):
    def __init__(self):
        super().__init__('ml_client')
        self.cli = self.create_client(XorRequest, 'xor_service')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = XorRequest.Request()
        
        self.send_request()
        
    def send_request(self):
        input_list = [random.randint(0, 1), random.randint(0, 1)]
        self.req.input = input_list
        
        self.future = self.cli.call_async(self.req)
        
        # Add a callback
        self.future.add_done_callback(self.response_callback)
        self.get_logger().info(f'Sending request: {input_list}')
        
    def response_callback(self, future):
        try:
            response = future.result()
            self.get_logger().info(f'Received response: Loss={response.loss}')
            self.send_request()
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')
            
def main():
    print('Hi from XOR_client.')

    rclpy.init()
    node = ml_client()
    rclpy.spin(node)
    
    # If we ever exit the spin loop
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
