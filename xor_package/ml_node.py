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

# message types
from std_msgs.msg import UInt8
from std_msgs.msg import UInt8MultiArray
from std_msgs.msg import Float32

# service types
from xor_package.srv import XorRequest

topic_result = 'topic_result'

if torch.cuda.is_available():
    device = torch.device('cuda')
else:    
    device = torch.device('cpu')

class XOR_model(nn.Module):
    def __init__(self):
        super(XOR_model, self).__init__()
        self.linear1 = nn.Linear(2, 4)
        self.linear15 = nn.Linear(4, 2)
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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        self.criterion = nn.BCELoss()
        self.srv = self.create_service(XorRequest, 'xor_service', self.xor_callback)
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


def main():
    print('Hi from XOR_server.')

    rclpy.init()
    node = ml_server()
    rclpy.spin(node)



if __name__ == '__main__':
    main()
