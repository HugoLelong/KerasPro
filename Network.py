import numpy as np
import Layer as ly
import Weight as wei
import Neuron as neu

class Network(object):
    """Class defining the network to train
        - inputSize the size of the input as (a,b,c)
        - layerList a list of all the layers of the network
        - learningRate the speed of learning
        - costFunction the cost function
        - weightDecay the parameter for the regularization which is not compulsory
    """
    
    def __init__(self, inputSize, learningRate, costFunction, weightDecay=None):
        self.inputSize=inputSize
        self.layerList=[]
        self.layerList.append(ly.Layer("input","sigmoid",self.inputSize[0]*self.inputSize[1]*self.inputSize[2],self))
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
   
    def addLayer(self, type, activationFunction, nbNeuron):
        myLayer=ly.Layer(type,activationFunction,nbNeuron,self)
        self.layerList.append(myLayer)
        n=len(self.layerList)
        if (type=="Dense"):
            previousNeurons=self.layerList[-2].getNeuronList()
            for i, previousNeuron in enumerate(myLayer.getNeuronList()):
                for j,currentNeuron in enumerate(previousNeurons):
                    currentNeuron.getWeightList().append(wei.Weight(previousNeuron,currentNeuron))

    '''def initializeWeights(self):
        """Create all the weights for a fully connected network"""
        for i in range(len(self.layerList)-1):
            for j,previousNeuron in enumerate(self.layerList[i].neuronList):
                for k,nextNeuron in enumerate(self.layerList[i+1].neuronList):
                    previousNeuron.getWeightList.append(wei.Weight(previousNeuron,nextNeuron))'''
    
    
    def alterLabels(self,labels):
        """Change the list of the labels into a matrix where matrix[i][j]=1 if the label of the ith example is j and 0 otherwise"""
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
        """Returns the cost calculated upon the crossentropy cost function, takes in argument the list of the labels"""
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
    
    def meanSquare(self,labels):
        """Return the cost calcuated upon the mean square cost function, takes in argument the list of the labels"""
        lastLayer=self.layerList[-1]
        neuronList=lastLayer.getNeuronList()
        aList=[]
        desiredOutput=self.alterLabels(labels)
        for neuron in neuronList:
            aList.append(neuron.getOutputNeuron())
        c=0
        for i in range(len(labels)):
            for j in range(len(neuronList)):
                c+=(desiredOutput[i][j]-aList[j])**2
        c=c/(2*len(neuronList))
        return(c)
        
    def toStringTab(self):
        """Return a string which represents the network"""
        sTab = self.getLayerList()[0].toStringTab()
        for i,layer in enumerate(self.getLayerList()):
            if(i!=0):
                sTabLayer = layer.toStringTab()
                for j in range(len(sTabLayer)):
                    if(j < len(sTab)):
                        sTab[j] += "     " + sTabLayer[j]
                    else:
                        sTab.append("              "*i + sTabLayer[j])
                if(len(sTabLayer) < len(sTab)):
                    for k in range(len(sTabLayer),len(sTab)):
                        sTab[k] += "              "
        return sTab
    
    def toString(self):
        sTab = self.toStringTab()
        s = ""
        for i in range(len(sTab)):
            s += sTab[i] + "\n\n"
        return s
    
    def firstLayerComputation(self,image_input):
        inputLayer=self.layerList[0]
        (a,b,c)=self.inputSize
        if (self.layerList[1].getType()=="Dense"):
            for i in range(a):
                for j in range(b):
                    for k in range(c):
                        self.layerList[0].getNeuronList()[k*a*b+i*a+j].setOutputNeuron(image_input[i][j][k])
    
    def feedforward(self, image_input):
        self.firstLayerComputation(image_input)
        for i,layer in enumerate(self.getLayerList()):
            if(i!=0):
                for j, neuron in enumerate(layer.getNeuronList()):
                    neuron.inputComputation()
                    neuron.outputComputation()
    
    
    
    
    
    
    