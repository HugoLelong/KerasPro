class Network:
    """Class defining the network to train
        - inputSize the size of the input
        - layerList a list of all the layers of the network
        - learningRate the speed of learning
        - costFunction the cost function
        - weightDecay the parameter for the regularization which is not compulsory
    """
    
    def __init__(self, inputSize, learningRate, costFunction, weightDecay=None):
        self.inputSize=inputSize
        self.layerList=[]
        self.learningRate=learningRate
        self.costFunction=costFunction
        self.weightecay=weightDecay
    
    