#!/usr/bin/env python3

from rclpy.node import Node
import rclpy
import torch
import torch.nn as nn
from torch import Tensor
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
import torchvision.transforms.functional as F
import torchvision
import torch.onnx

import os
from ament_index_python.packages import get_package_share_directory

# message types
from std_msgs.msg import UInt8
from std_msgs.msg import UInt8MultiArray
from std_msgs.msg import Float32

# service types
from xor_package.srv import XorRequest
from xor_package.srv import SaveModel

topic_result = 'topic_result'

if torch.cuda.is_available():
    device = torch.device('cuda')
else:    
    device = torch.device('cpu')

class XOR_model(nn.Module):
    def __init__(self):
        super(XOR_model, self).__init__()
        self.linear1 = nn.Linear(2, 8)
        self.linear15 = nn.Linear(8, 2)
        self.linear2 = nn.Linear(2, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear15(x))
        x = torch.sigmoid(self.linear2(x))
        return x

class ml_server(Node):
    def __init__(self):
        super().__init__('ml_server')
        self.model = XOR_model()
        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.criterion = nn.BCELoss()
        self.srv = self.create_service(XorRequest, 'xor_service', self.xor_callback)
        self.save_model_srv = self.create_service(SaveModel, 'save_model', self.save_model_callback)
        self.publisher_ = self.create_publisher(UInt8, topic_result, 10)

    def xor_callback(self, request, response):
            if len(request.input) != 2:
                self.get_logger().error("Input must have exactly 2 elements")
                return response
            
            input_tensor = torch.tensor([request.input], dtype=torch.float32).to(device)
            
            target_value = request.input[0] ^ request.input[1]
            target = torch.tensor([[target_value]], dtype=torch.float32).to(device)

            ar = self.predict_and_backward(input_tensor, target)
            response.loss = ar[0] 

            result = 1 if ar[1] >= 0.5 else 0
            
            msg = UInt8()
            msg.data = result
            self.publisher_.publish(msg)

            self.get_logger().info(f"Input: {request.input}, Predicted: {result}, Loss: {ar[0]}")
                
            return response
        
    def predict_and_backward (self, input_tensor: Tensor, target: Tensor) -> list[float]:
        output = self.model(input_tensor)
        loss = self.criterion(output, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return [loss.item(), output.item()]
    
    def save_model_callback(self, request, response):
        package_share_directory = get_package_share_directory('xor_package')
        models_dir = os.path.join(package_share_directory, 'models')
        
        # Ensure the models directory exists
        os.makedirs(models_dir, exist_ok=True)
        
        model_path = os.path.join(models_dir, 'xor_model.pth')
        torch.save(self.model.state_dict(), model_path)
        self.get_logger().info(f"Model saved to: {model_path}")
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
