# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 14:21:25 2018

@author: Hugo
"""

def test():
    network = Network([2,1,1],0.5,"meanSquare")
    network.addLayer("Dense","sigmoid",2)
    network.addLayer("Dense","sigmoid",2)
    network.getLayerList()[0].getNeuronList()[0].getWeightList()[0].setValue(0.15)
    network.getLayerList()[0].getNeuronList()[0].getWeightList()[1].setValue(0.2)
    network.getLayerList()[0].getNeuronList()[1].getWeightList()[0].setValue(0.25)
    network.getLayerList()[0].getNeuronList()[1].getWeightList()[1].setValue(0.3)
    network.getLayerList()[1].getNeuronList()[0].getWeightList()[0].setValue(0.4)
    network.getLayerList()[1].getNeuronList()[0].getWeightList()[1].setValue(0.45)
    network.getLayerList()[1].getNeuronList()[1].getWeightList()[0].setValue(0.5)
    network.getLayerList()[1].getNeuronList()[1].getWeightList()[1].setValue(0.55)
    network.getLayerList()[1].getNeuronList()[0].setBias(0.35)
    network.getLayerList()[1].getNeuronList()[1].setBias(0.35)
    network.getLayerList()[2].getNeuronList()[0].setBias(0.6)
    network.getLayerList()[2].getNeuronList()[1].setBias(0.6)
    network.feedforward([[[0.05]],[[0.1]]])    
    print(network.toString())
    network.backpropagate([[[[0.05]],[[0.1]]]],[[0.01, 0.99]])
    