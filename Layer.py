class Layer:
    """Class defining the layers found in a network:
        - its type (dense layer, Convolutional layer ...) called type
        - a list of Neurons called neuronList
        - a number a neuron called nNeuron
        - its activation function called activationFun """
    
    def __init__(self, type, activationFunction, nNeuron):
        """neuronList is initialized as void and activationFunction is activationFun"""
        self.type=type
        self.neuronList=[]
        self.activationFun=activationFunction
        self.nNeuron=nNeuron
        