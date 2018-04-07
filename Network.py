class Network(object):
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
    
    def getLayerList(self):
        return self.layerList
    
    def addLayer(self,nbNeuron):
        self.layerList.append(Layer(1,1,nbNeuron,self))
    
    def initializeWeights(self):
        for i in range(len(self.layerList)-1):
            for j,previousNeuron in enumerate(self.layerList[i].neuronList):
                for k,nextNeuron in enumerate(self.layerList[i+1].neuronList):
                    previousNeuron.weightList.append(Weight(previousNeuron,nextNeuron))