# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 15:33:19 2018

@author: Hugo
"""

class Neuron:
    """Class which defines a neuron in a neural network. It is composed of
    - the layer in which is this neuron
    - the index of the neuron in this layer
    - its bias
    - the list of its weights between this neuron and neurons in the next layer called weightList"""
    
    def __init__(self, layer, index, bias):
        """The list of weights is initialized with an empty list, 
        we will create a weigth between each neuron later"""
        self.layer = layer
        self.index = index
        self.bias = bias
        self.weightList = []