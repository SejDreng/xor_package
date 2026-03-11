#!/usr/bin/env python3

from rclpy.node import Node
import rclpy

import numpy as np
import tensorflow as tf

# Running tflite model
# import torch
# import torch.nn as nn
# from torch import Tensor
# import torchvision.transforms as transforms
# from torchvision.transforms import ToTensor
# import torchvision.transforms.functional as F
# import torchvision
# import torch.onnx

import os
from ament_index_python.packages import get_package_share_directory

# message types
from std_msgs.msg import UInt8

# service types
from xor_package.srv import XorRequest

topic_result = 'topic_result'

# original pytorch model for reference. 
# Converted with:
# onnx2tf -i xor_model.onnx -o saved_model -oiqt \
#   -cind x Datasets/calibration_data.npy [0.0] [1.0] \
#   -iqd int8 -oqd int8 

# REMEMBER TO PREFORM QUANTIZATION

# class XOR_model(nn.Module):
#     def __init__(self):
#         super(XOR_model, self).__init__()
#         self.linear1 = nn.Linear(2, 8)
#         self.linear15 = nn.Linear(8, 2)
#         self.linear2 = nn.Linear(2, 1)

#     def forward(self, x: Tensor) -> Tensor:
#         x = torch.relu(self.linear1(x))
#         x = torch.relu(self.linear15(x))
#         x = torch.sigmoid(self.linear2(x))
#         return x

class ml_server(Node):
    def __init__(self):
        super().__init__('ml_server')
        package_share_dir = get_package_share_directory('xor_package')

        # This matches the folder structure created by the 'install' command in CMake
        self.model_path = os.path.join(
            package_share_dir, 
            'models', 
            'saved_model', 
            'xor_model_full_integer_quant.tflite'
        )

        self.get_logger().info(f"Loading model from: {self.model_path}")
        model_path = self.model_path
        
        # Should be this on voxl2 if we copy the model to /usr/bin/dnn/ and specify it in the .config, i will instead use system-level NNAPI directly--- IGNORE ---:
        # model_path = "/usr/bin/dnn/xor_model_full_integer_quant.tflite"
        
        try:
            # This invokes the system-level NNAPI, which automatically 
            # finds the correct DSP/GPU accelerator for the QRB5165
            nnapi_delegate = tf.lite.load_delegate("libnnapi_delegate.so")
            delegates = [nnapi_delegate]
            self.get_logger().info("NNAPI Delegate loaded successfully!")
        except Exception as e:
            self.get_logger().warn(f"Could not load NNAPI: {e}. Falling back to CPU.")
            delegates = []
                
        self.interpreter = tf.lite.Interpreter(model_path=model_path, experimental_delegates=delegates)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Get quantization parameters for Full INT8
        self.input_scale, self.input_zero_point = self.input_details[0]['quantization']
        self.output_scale, self.output_zero_point = self.output_details[0]['quantization']

        self.srv = self.create_service(XorRequest, 'xor_service', self.xor_callback)
        self.publisher_ = self.create_publisher(UInt8, 'topic_result', 10)
        self.get_logger().info("TFLite Model loaded and ready.")

    def xor_callback(self, request, response):
        if len(request.input) != 2:
            self.get_logger().error("Input must have exactly 2 elements")
            return response
        
        input_data = np.array([request.input], dtype=np.float32)
        input_tensor = (input_data / self.input_scale + self.input_zero_point).astype(np.int8)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        ar = (output_data.astype(np.float32) - self.output_zero_point) * self.output_scale
    
        result = 1 if ar >= 0.5 else 0
        
        msg = UInt8()
        msg.data = result
        self.publisher_.publish(msg)

        self.get_logger().info(f"Input: {request.input}, Predicted: {result}")
        response.success = True
        return response

def main():
    print('Hi from XOR_server.')

    rclpy.init()
    node = ml_server()
    rclpy.spin(node)

    # If we ever exit the spin loop
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
