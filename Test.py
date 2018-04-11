# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 14:21:25 2018

@author: Hugo
"""

def test():
    network = Network([2,1,1],1,"meanSquare")
    network.addLayer("Dense","sigmoid",2)
    network.addLayer("Dense","sigmoid",2)
    network.getLayerList()[0].getNeuronList()[0].getWeightList()[0].setValue(0.75)
    network.getLayerList()[0].getNeuronList()[0].getWeightList()[1].setValue(0.45)
    network.getLayerList()[0].getNeuronList()[1].getWeightList()[0].setValue(0.2)
    network.getLayerList()[0].getNeuronList()[1].getWeightList()[1].setValue(0.5)
    network.getLayerList()[1].getNeuronList()[0].getWeightList()[0].setValue(0.6)
    network.getLayerList()[1].getNeuronList()[0].getWeightList()[1].setValue(0.7)
    network.getLayerList()[1].getNeuronList()[1].getWeightList()[0].setValue(0.1)
    network.getLayerList()[1].getNeuronList()[1].getWeightList()[1].setValue(0.4)
    network.getLayerList()[1].getNeuronList()[0].setBias(0.3)
    network.getLayerList()[1].getNeuronList()[1].setBias(-0.5)
    network.getLayerList()[2].getNeuronList()[0].setBias(0.05)
    network.getLayerList()[2].getNeuronList()[1].setBias(-0.5)
    network.feedforward([[[0.4]],[[-0.45]]])    
    print(network.toString())
    network.backpropagate([[[[0.4]],[[-0.45]]]],[[0.35,-0.5]])
    