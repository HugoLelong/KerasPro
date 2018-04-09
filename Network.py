import numpy as np
import Layer as ly
import Weight as wei
import Neuron as neu
from math import floor
import random

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
        '''Adds the layer created using the given parameters, creates also the weigths linking the added layer to the previous one'''
        myLayer=ly.Layer(type,activationFunction,nbNeuron,self)
        self.layerList.append(myLayer)
        if (type=="Dense"):
            previousNeurons=self.layerList[-2].getNeuronList()
            for i, previousNeuron in enumerate(previousNeurons):
                for j,nextNeuron in enumerate(myLayer.getNeuronList()):
                    previousNeuron.getWeightList().append(wei.Weight(previousNeuron,nextNeuron))

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
        """Compute all the weights from the image given in parameter as image_input, the weight is the value of the pixel (R,V or B)"""
        (a,b,c)=self.inputSize
        if (self.layerList[1].getType()=="Dense"):
            for i in range(a):
                for j in range(b):
                    for k in range(c):
                        print(i,j,k)
                        self.layerList[0].getNeuronList()[k*a*b+i*b+j].setOutputNeuron(image_input[i][j][k])
    
    def feedforward(self, image_input):
        self.firstLayerComputation(image_input)
        for i,layer in enumerate(self.getLayerList()):
            if(i!=0):
                for j, neuron in enumerate(layer.getNeuronList()):
                    neuron.inputComputation()
                    neuron.outputComputation()
    
    def backpropagate(self):
        """Function to complete to make the backpropagation of the information, the cost function are coded above (meanSquare and crossentropy), if you want to add parameters think to add them in train too"""
        pass
    
    def prediction(self,image):
        """Function which computes the prediction of class the image was in given the image (parameter image). The output is the index of the neuron which gave the greatest value. If the labels where coded with a serie of following numbers beginning by 0, it is also the class of the image."""
        lastLayer=self.layerList[-1]
        outputValues=[]
        for neuron in lastLayer.getNeuronList():
            outputValues.append(neuron.getOutputNeuron())
        M=0
        indexM=-1
        for j,value in enumerate(outputValues):
            if value>M:
                M=value
                indexM=j
        return(indexM)
    
    def train(self,imageTrainSet,labelTrainSet,nbEpochs,batchSize,validationData=None):
        """Function used to train the network, the parameters are the total list of images, the LIST of the labels, the number of epochs, the batch size (must be less than the number of train images) and the validation data as a tuple (validationImages, validationLabels). If not put, it is None and not taken into account. The outputs are the number of true predictions (for the train set and the validation set) per epoch in two different lists."""
        imageBatchSize=[]
        labelBatchSize=[]
        trueGuessList=[]
        trueGuessValList=[]
        for i in range(nbEpochs):
            while (len(imageBatchSize)<batchSize):
                n=random.randint(0,len(imageTrainSet)-1)
                if imageTrainSet[n] not in imageBatchSize:
                    imageBatchSize.append(imageTrainSet[n])
                    labelBatchSize.append(labelTrainSet[n])
            trueGuess=0
            for j,image in enumerate(imageBatchSize):
                print(image)
                self.feedforward(image)
                if self.prediction(image)==labelBatchSize[j]:
                    trueGuess+=1
            self.backpropagate()
            trueGuessList.append(trueGuess)
            if(validationData!=None):
                trueGuessVal=0
                (imageValidationList,labelValidationList)=validationData
                for k,imageVal in enumerate(imageValidation):
                    self.feedforward(imageVal)
                    if(self.prediction(imageVal)==labelValidationList[k]):
                        trueGuessVal+=1
                trueGuessValList.append(trueGuessVal)
        return(trueGuessList,trueGuessValList)
            
        
        
    
    
    
    
    