# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 01:09:05 2020

@author: Franz
"""

import numpy as np
from datetime import datetime

def morse(D,r0,gamma,r):
    chi = np.exp(-gamma/2*((r/r0)-1))
    energy = D*(chi**2 - 2*chi)
    return energy

def derivativeMorseD(r0,gamma,r):
    chi = np.exp(-gamma/2*((r/r0)-1))
    derivativeEnergy = (chi**2 - 2*chi)
    return derivativeEnergy

def derivativeMorseGamma(D,r0,gamma,r):
    chi = np.exp(-gamma/2*((r/r0)-1))
    derivativeEnergy = D*((r/r0)-1)*(chi-chi**2)
    return derivativeEnergy

def derivativeMorseR0(D,r0,gamma,r):
    chi = np.exp(-gamma/2*((r/r0)-1))
    derivativeEnergy = (D*gamma*r/r0**2)*(chi**2 - chi)
    return derivativeEnergy

def getEnergies(configs,parameters,distances):
    numberOfInteractions = distances.shape[0]
    interactionEnergy = np.zeros((numberOfInteractions))
    for i in range(numberOfInteractions):
        interactionType = int(distances[i][0])
        interactionEnergy[i] = morse(parameters[interactionType][0],
                                    parameters[interactionType][1],
                                    parameters[interactionType][2],
                                    distances[i][1])

    energies = np.zeros((configs))
    intersPerConfig = (numberOfInteractions/configs)
    for i in range(configs):
        lowerIndex = int(i*intersPerConfig)
        upperIndex = int(lowerIndex + intersPerConfig)
        energies[i] = np.sum(interactionEnergy[lowerIndex:upperIndex])
    return energies

def getEnergyGradients(interactionType,parameterType,parameters,distances):
    numberOfInteractions = distances.shape[0]
    interactionGradient = np.zeros((numberOfInteractions))
    for i in range(numberOfInteractions):
        if interactionType == int(distances[i][0]):
            if parameterType == 0:
                interactionGradient[i] = derivativeMorseD(parameters[interactionType][1],
                                                            parameters[interactionType][2],
                                                            distances[i][1])
            elif parameterType == 1:
                interactionGradient[i] = derivativeMorseR0(parameters[interactionType][0],
                                                            parameters[interactionType][1],
                                                            parameters[interactionType][2],
                                                            distances[i][1])
            elif parameterType == 2:
                interactionGradient[i] = derivativeMorseGamma(parameters[interactionType][0],
                                                            parameters[interactionType][1],
                                                            parameters[interactionType][2],
                                                            distances[i][1])

    return np.sum(interactionGradient)

dataset = open("dataset.txt","r")

configs = 13

optimizerNumberOfSteps = 5000
energyWeight = 0.1  # the lower the value the more strict you want to be in fit                                

learningRate = 0.001
numberOfEpochs = 1000

# energies must be in kcal/mol
dftEnergies = np.array([  0.038,  5.507, 12.822, 3.400, -0.961, -0.659,
                         -0.258, -2.311, -3.360,-3.821, -4.211, -3.161,
                         -0.178 ])

# do not modify below this

data = []
for line in dataset:
        data.append(line.split())

dataset.close()

inters = len(data)
intersPerConfig = int(inters/configs)

distances = np.zeros((inters,2))
for i in range(inters):
    distances[i][0] = int(data[i][1])
    distances[i][1] = float(data[i][2])

#definitions for pairType: 0 = Pt--H, 1 = Pt--C3, 2 = Pt--C2
#   3 = Pt--O, 4 = Pt--H_O
parameters = np.zeros((5,3))

#Initial guess
#parameters[interactionType][0:3], 0 = D, 1 = r0, 2 = gamma
parameters[0][0:3] = [1.364,3.054,9.830]
parameters[1][0:3] = [0.338,4.476,13.789]
parameters[2][0:3] = [0.338,4.476,13.789]
parameters[3][0:3] = [2.500,2.416,11.648]
parameters[4][0:3] = [1.364,3.054,9.830]

energies = getEnergies(configs,parameters,distances)
                       
lossFunction = np.sum(np.square(dftEnergies - energies))/configs
print("Initial loss function {}".format(lossFunction))

# Stochastic gradient descent
lossFunctionGradient = np.zeros((5,3))
for epoch in range(numberOfEpochs):
    randomConfig = np.random.randint(configs)
    configDistances = distances[intersPerConfig*randomConfig:intersPerConfig*(1+randomConfig)][:]

    for interactionType in range(5):
        for parameterType in range(3):
            energyGradients = getEnergyGradients(interactionType,parameterType,parameters,configDistances)
            lossFunctionGradient[interactionType][parameterType] = -2.0*(dftEnergies[randomConfig] - energies[randomConfig])*energyGradients

    newParameters = parameters - learningRate*lossFunctionGradient

    #physical constraints
    if any(newParameters[:][1] <= 0.0):
        print("epoch skipped due to unphysical r0: negative or zero")
        continue
    if any(newParameters[:][2] < 0.0):
        print("epoch skipped due to negative gamma")
        continue
    
    parameters = newParameters
    energies = getEnergies(configs,parameters,distances)

    oldLossFunction = lossFunction
    lossFunction = np.sum(np.square(dftEnergies - energies))/configs
    print("Current loss function {}".format(lossFunction))

    if lossFunction < oldLossFunction:
        bestLossFunction = lossFunction
        bestParameters = parameters
        bestEnergies = energies

print(parameters)
print(energies)
print(lossFunction)

print("---------------")
print(bestParameters)
print(bestEnergies)
print(bestLossFunction)
