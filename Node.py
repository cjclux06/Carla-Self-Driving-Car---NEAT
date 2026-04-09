import math as Math
from enum import Enum

class NodeType(Enum):
    INPUT = "INPUT"
    HIDDEN = "HIDDEN"
    OUTPUT = "OUTPUT"

class Node():

    def __init__(self, node=None, type=None, innovation=None):
        self.value = 0
        self.type = type
        self.innovation = innovation

        if node != None:
            self.type = node.type
            self.innovation = node.innovation

    def activation_function_sigmoid(self):
        self.value =  1 / (1 + Math.exp(-self.value));  #Sigmoid function

    def activation_function_relu(self):
        self.value = self.value if self.value > 0 else 0.05 * self.value