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
        self.layerIndex=len(self.network.getLayerList())-1
        for j in range(self.nbNeuron):
            myNeuron=Neuron(self,j)
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