# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 14:21:25 2018

@author: Hugo
"""

def test():
    network = Network([5,1,1],1,1)
    network.addLayer("Dense","sigmoid",7)
    network.addLayer("Dense","sigmoid",10)
    network.feedforward([[[0.2]],[[1.4]],[[2.1]],[[0.7]],[[0.9]]])
    
    print(network.toString())