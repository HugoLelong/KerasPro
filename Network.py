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
        self.weightDecay=weightDecay
    
    def getLayerList(self):
        return self.layerList
    
    def getInputSize(self):
        return self.inputSize
    
    def getLearningRate(self):
        return self.learningRate
    
    def getCostFunction(self):
        return self.costFunction
    
    def getWeightDecay(self):
        return self.weightDecay
    
    def addLayer(self,nbNeuron):
        self.getLayerList().append(Layer(1,1,nbNeuron,self))
    
    def initializeWeights(self):
        """Create all the weights for a fully connected network"""
        for i in range(len(self.layerList)-1):
            for j,previousNeuron in enumerate(self.layerList[i].neuronList):
                for k,nextNeuron in enumerate(self.layerList[i+1].neuronList):
                    previousNeuron.weightList.append(Weight(previousNeuron,nextNeuron))
    
    
    def alterLabels(labels):
        l=[]
        last
        
        
    def crossentropy(self,labels):
        lastLayer=layerList[-1]
        neuronList=lastLayer.getNeuronList()
        aList=[]
        desi
        for i,neuron in enumerate(neuronList):
            aList.append(neuron.getOutputNeuron())
            for n in range(len(labels)):
                desiredOutput
        
            
        