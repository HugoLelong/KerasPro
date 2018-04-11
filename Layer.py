import numpy as np
from math import floor
import Weight as wei
import Neuron as neu
import Network as net
from decimal import *

class Layer:
    """Class defining the layers found in a network:
        - its type (dense layer, Convolutional layer ...) called type
        - a list of Neurons called neuronList
        - a number a neuron called nbNeuron
        - its activation function called activationFun
        - a network (in which this layer is) 
        - an index being the index of the layer in its network"""
    
    def __init__(self, typeLayer, activationFunction, nbNeuron, network):
        """neuronList is initialized as void, activationFunction is activationFun, the index is set refering to network, the first layer has index 0"""
        self.type=typeLayer
        self.neuronList=[]
        self.activationFun=activationFunction
        self.nbNeuron=nbNeuron
        self.network=network
        self.layerIndex=len(self.network.getLayerList())
        for j in range(self.nbNeuron):
            myNeuron=neu.Neuron(self,j)
            self.neuronList.append(myNeuron)
    
    def getType(self):
        return self.type
    
    def getNeuronList(self):
        return self.neuronList
    
    def getActivationFunction(self):
        return self.activationFun
    
    def getNetwork(self):
        return self.network
    
    def getLayerIndex(self):
        return self.layerIndex
    
    def getNbNeuron(self):
        return self.nbNeuron
    
    def sigmoid(cls,x):
        return(1/(1+np.exp(-x)))
    
    sigmoid=classmethod(sigmoid)
    
    def sigmoidprime(cls,x):
        s=Layer.sigmoid(x)
        return (s*(1-s))
    
    sigmoidprime=classmethod(sigmoidprime)
    
    def relu(cls,x):
        return max(0,x)
    
    relu=classmethod(relu)
    
    def softmaxResults(self):
        """Return a list of all the softmax probabilities"""
        exponentialSum=0
        exponentialList=[]
        valueList=[]
        for i in range(self.nbNeuron):
            value=np.exp(self.neuronList[i].getInputNeuron())
            exponentialSum+=value
            valueList.append(value)
        exponentialList=[valueList[i]/exponentialSum for i in range(self.nbNeuron)]
        return(exponentialList)

    def toStringTab(self):
        s = []
        for i,neuron in enumerate(self.getNeuronList()):
            s.append(str(Decimal(neuron.getInputNeuron()).quantize(Decimal('.01'), rounding=ROUND_HALF_UP)) + " - " + str(Decimal(neuron.getOutputNeuron()).quantize(Decimal('.01'), rounding=ROUND_HALF_UP)))
        return s
        
    def toString(self):
        sTab = self.toStringTab()
        s = ""
        for i in range(len(sTab)):
            s += sTab[i] + "\n\n"
        return s
        
        
        
        
        