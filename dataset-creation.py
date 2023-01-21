# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 10:52:25 2020

@author: Franz
"""

import re
import numpy as np

def getDistances(filename):
    lammpsfile = open(filename,'r')
    fulltext = lammpsfile.read()
    lammpsfile.close()

    configBlock = re.search("Atoms.*Bond",fulltext,re.DOTALL).group()
    lines = configBlock.splitlines()

    atomPosition = [ lines[2].split()[4],
                     lines[2].split()[5],
                     lines[2].split()[6] ]

    atomPositionList = []
    for i in range(60):
        atomPosition = [ float(lines[2+i].split()[4]),
                         float(lines[2+i].split()[5]),
                         float(lines[2+i].split()[6]) ]
        atomPositionList.append(atomPosition)

    configuration = np.array(atomPositionList)

    ptslab = configuration[0:48]
    ipa = configuration[48:60]
    
    pairInfo = []

    for ptposition in ptslab:
        ipaAtomCount = 0
        for ipaposition in ipa:
            #definitions for pairType: 0 = Pt--H, 1 = Pt--CH3, 2 = Pt--CHO
            #   3 = Pt--O, 4 = Pt--H_O
            pairType = 0
            if ipaAtomCount == 0 or ipaAtomCount == 5:
                pairType = 1
            if ipaAtomCount == 1:
                pairType = 2
            if ipaAtomCount == 9:
                pairType = 3
            if ipaAtomCount == 11:
                pairType = 4
                
            pairInfo.append([filename,
                             pairType,
                             np.linalg.norm(ptposition-ipaposition)])
            
            ipaAtomCount = ipaAtomCount + 1
            
    return pairInfo


#only modify numberOfConfigurations variable

dataset = open("dataset.txt",'w')
numberOfConfigurations = 13
for i in range(numberOfConfigurations):
    filename = "mdpoint-" + str(i+1) + ".data" #you need your files named mdpoint-1.data, mdpoint-2.data, ...
    data = getDistances(filename)
 
    for j in range(len(data)):
        dataset.write("{}  {}  {} \n".format(data[j][0],data[j][1],data[j][2]))
    
dataset.close()
    