import Neuron

class Layer:
    """Class defining the layers found in a network:
        - its type (dense layer, Convolutional layer ...) called type
        - a list of Neurons called neuronList
        - a number a neuron called nNeuron
        - its activation function called activationFun
        - a network (in which this layer is) 
        - an index being the index of the layer in its network"""
    
    def __init__(self, type, activationFunction, nNeuron, network):
        """neuronList is initialized as void, activationFunction is activationFun, the index is set refering to network"""
        self.type=type
        self.neuronList=[]
        self.activationFun=activationFunction
        self.nNeuron=nNeuron
        self.network=network
        self.index=len(network.getLayerList(self.network))
        for j in range(self.nNeuron):
            myNeuron=Neuron(self,j)
            self.neuronList.append(myNeuron)