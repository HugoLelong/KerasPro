import numpy as np
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
        self.layerList.append(Layer(1,2,self.inputSize,self))
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
                    previousNeuron.getWeightList.append(Weight(previousNeuron,nextNeuron))
    
    
    def alterLabels(self,labels):
        """Change the list of the labeld=s in a matrix where matrix[i][j]=1 if the label of the ith example is j and 0 otherwise"""
        l=[]
        nbLastNeurons=len(self.layerList[-1].getNeuronList())
        for i in range(len(labels)):
            l.append([])
            for j in range(nbLastNeurons):
                if (j==labels[i]):
                    l[i].append(1)
                else:
                    l[i].append(0)
        return(l)
    
        
    def crossentropy(self,labels):
        """Return the cost calculated upon the crossentropy cost function, take in argument the list of the labels"""
        lastLayer=self.layerList[-1]
        neuronList=lastLayer.getNeuronList()
        aList=[]
        desiredOutput=self.alterLabels(labels)
        c=0
        for neuron in neuronList:
           aList.append(neuron.getOutputNeuron())
        for i in range(len(labels)):
            for j in range(len(neuronList)):
                y=desiredOutput[i][j]
                a=aList[j]
                c+=y*np.log(a)+(1-y)*np.log(1-a)
        return(c/len(labels))
        
            
        