from random import gauss
import Layer as ly
import Network as net
import Weight as wei

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
    
    def __init__(self, layer, index):
        """The list of weights is initialized with an empty list, 
        we will create a weigth between each neuron later, the bias follows a gaussian lawS"""
        self.layer = layer
        self.index = index
        self.bias = gauss(0,1)
        self.weightList = []
        self.inputNeuron = 0
        self.outputNeuron = 0
    
    def getLayer(self):
        return self.layer
    
    def getIndex(self):
        return self.index
    
    def getBias(self):
        return self.bias
    
    def getWeightList(self):
        return self.weightList
    
    def getInputNeuron(self):
        return self.inputNeuron
    
    def getOutputNeuron(self):
        return self.outputNeuron
    
    def setBias(self, newBias):
        self.bias=newBias
        
    def setInputNeuron(self, value):
        self.inputNeuron = value

    def setOutputNeuron(self, value):
        self.outputNeuron = value
        
    def inputNeuronComputation(self):
        """Compute z = wx + b if this neuron is not in the input layer
        where w is the list of weights, x the list of input in the previous layer
        and b is the bias"""
        if(self.layer.getLayerIndex() != 0):
            value = self.getBias()
            for i,previousNeuron in enumerate(self.layer.getNetwork().getLayerList()[self.layer.getLayerIndex()-1].getNeuronList()):
                weight = 0
                for j, objectWeight in enumerate(previousNeuron.getWeightList()):
                    if(objectWeight.getNextNeuron() == self):
                        weight = objectWeight.getValue()
                value += previousNeuron.getInputNeuron() * weight
            self.setInputNeuron(value)
    
    def outputComputation(self):
        activationFun=self.layer.getActivationFunction()
        if(activationFun=="sigmoid"):
            self.outputNeuron=ly.Layer.sigmoid(self.inputNeuron)
        elif(activationFun=="relu"):
            self.outputNeuron=ly.Layer.relu(self.inputNeuron)
        elif(activationFun=="softmax"):
            self.outputNeuron=self.layer.softmaxResults()[self.index]
        return(self.outputNeuron)
        
        
        
        
        
        
        
        