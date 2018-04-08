# -*- coding: utf-8 -*-

"""
Created on Sat Apr  7 15:57:02 2018

@author: Hugo
"""
from random import gauss
import Neuron as neu
import Network as net
import Layer as ly

class Weight:
    """Class which defines the link between two neurons. It is composed of:
    - a value
    - the neuron in the previous layer
    - the neuron in the next layer"""
    
    def __init__(self, previousNeuron, nextNeuron):
        """The value of the weight is going to be initialized randomly"""
        self.previousNeuron = previousNeuron
        self.nextNeuron = nextNeuron
        self.value = gauss(0,1)
    
    def getValue(self):
        return self.value
    
    def getNextNeuron(self):
        return(self.nextNeuron)
    
    def getPreviousNeuron(self):
        return self.previousNeuron
    
    def setValue(self, newValue):
        self.value=newValue