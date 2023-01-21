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

dataset = open("dataset.txt","r")

configs = 13

optimizerNumberOfSteps = 5000
energyWeight = 0.1  # the lower the value the more strict you want to be in fit                                

learningRate = 0.01

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
                       
configErrorFunction = np.square((energies - dftEnergies)/energyWeight)
errorFunction = np.sum(configErrorFunction)

lossFunction = np.sum(np.square(dftEnergies - energies))/configs
print("Initial loss function")
print(lossFunction)

# Stochastic gradient descent

randomConfig = np.random.randint(configs)

interactionGradient = np.zeros((intersPerConfig))
for i in range(intersPerConfig):
    tempIndex = i + intersPerConfig*randomConfig
    interactionType = int(distances[tempIndex][0])
    if interactionType == 0:
        interactionGradient[i] = derivativeMorseD(parameters[0][1],
                                                    parameters[0][2],
                                                    distances[tempIndex][1])

lossFunctionGradient = -2.0*(dftEnergies[randomConfig] - energies[randomConfig])*np.sum(interactionGradient)
print(randomConfig,lossFunctionGradient)


parameters[0][0] = parameters[0][0] - learningRate*lossFunctionGradient
print(parameters[0][0])

energies = getEnergies(configs,parameters,distances)
                       
configErrorFunction = np.square((energies - dftEnergies)/energyWeight)
errorFunction = np.sum(configErrorFunction)

lossFunction = np.sum(np.square(dftEnergies - energies))/configs
print("Final loss function")
print(lossFunction)

exit()

oldErrorFunction = errorFunction
bestErrorFunction = errorFunction
bestParameters = np.zeros((5,3))
bestEnergies = np.zeros((configs))
success = 0
step = 0.001
for j in range(optimizerNumberOfSteps):
    randomIndex = np.random.randint(low=0,high=5)
    
    randomParam = np.random.randint(low=0,high=3)
    
    oldparameter = parameters[randomIndex][randomParam]
        
    if np.random.random() < 0.5:
        tryparameter = oldparameter + step
    else:
        tryparameter = oldparameter - step
        
    if randomParam == 0 and tryparameter <= 1e-5:
        tryparameter = 10.0*np.random.random_sample()
        
    if randomParam == 1 and tryparameter <= 0.5:
        tryparameter = 4.0*np.random.random_sample() + 1.0
        
    parameters[randomIndex][randomParam] = tryparameter

    interactionEnergy = np.zeros((inters))
    for i in range(inters):
        interactionType = int(distances[i][0])
        interactionEnergy[i] = morse(parameters[interactionType][0],
                                     parameters[interactionType][1],
                                     parameters[interactionType][2],
                                     distances[i][1])
    
    energies = np.zeros((configs))
    for i in range(configs):
        lowerIndex = int(i*intersPerConfig)
        upperIndex = int(lowerIndex + intersPerConfig)
        energies[i] = np.sum(interactionEnergy[lowerIndex:upperIndex])
                           
    configErrorFunction = np.square((energies - dftEnergies)/energyWeight)
    errorFunction = np.sum(configErrorFunction)
    
    deltaErrorFunction = errorFunction - oldErrorFunction
    if deltaErrorFunction < 0:
        oldErrorFunction = errorFunction
        success = success + 1
    else:
        conditionalNumber = np.exp(-deltaErrorFunction/20)
        if np.random.random() <= conditionalNumber:
            oldErrorFunction = errorFunction
            success = success + 1
        else:
            parameters[randomIndex][randomParam] = oldparameter
        
    if errorFunction > oldErrorFunction:
        parameters[randomIndex][randomParam] = oldparameter
    else:
        oldErrorFunction = errorFunction
        success = success + 1
    
    if success/(j+1) > 0.4:
        step = step*1.05
    else:
        step = step*0.95
        
    if step > 0.2 or step < 0.0001:
        step = 0.001
        
    if errorFunction < bestErrorFunction:
        bestErrorFunction = errorFunction
        bestParameters = parameters
        bestEnergies = energies
        
        results = open("results.txt","w")
        timeNow = datetime.now()
        results.write("{} \n".format(timeNow))
        results.write("error Function {}\n".format(bestErrorFunction))
        results.write("Parameters \n")
        outBestParameters = np.array2string(bestParameters, precision=5)
        results.write("{} \n".format(outBestParameters))
        results.write("DFT reference energies \n")
        outRefDFTEnergies = np.array2string(dftEnergies, precision=3)
        results.write("{} \n".format(outRefDFTEnergies))
        results.write("Force field energies \n")
        outBestEnergies = np.array2string(bestEnergies, precision=3)
        results.write("{} \n".format(outBestEnergies))
        
        rmsd = np.sqrt(np.sum(np.square(bestEnergies - dftEnergies))/configs)
        results.write("RMSD of energies (less is better) {}".format(rmsd))
        results.close()
        
    if j%100 == 0:
        if j%1000 == 0:
            print("step, current EF, previous EF, step size")
            
        print(j,oldErrorFunction,errorFunction,step)

print("Last Error Function")
print(errorFunction)
print("Energies from FF last parameters")
print(energies)
print("Energies from DFT (reference)")
print(dftEnergies)
print("New parameters of the FF")
print(parameters)