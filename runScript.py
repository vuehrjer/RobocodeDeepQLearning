import random
import math
import subprocess
import os, sys, time
import numpy as np
from copy import deepcopy
import fileinput
robocodePath = 'C:/robocode/'
dataPath = os.path.dirname(sys.argv[0]) + '/out/production/RobocodeDeepQLearning/ARL/Rl_nn.data/'
subprocesses = []

topParentPercent = 0.8
mutationChance = 0.8
inputNeurons = 8
hiddenLayerNeurons = 10
outputNeurons = 1
populationSize = 10
randomWeightStandardDeviation = 20
hyperparamAmount = 11
hyperparamStandardDeviation = 10
# hyperparams:
# alpha,
# gamma,
# rho,
# epsilon
# hitBulletReward,
# hitByBulletReward,
# hitRobotReward,
# hitwallReward,
# onDeathReward,
# onWinReward,
# hiddenLayerNeurons

class Robot:
    hyperparams = [0] * hyperparamAmount
    fitness = 0
    id = 0
    def __init__(self, id):
        self.id = id

    def loadHyperparams(self):
        self.hyperparams = loadHyperparams( str(self.id) + 'hyperparams.txt')

    def saveHyperparams(self):
        saveHyperparams(self.hyperparams, str(self.id) + 'hyperparams.txt')

    def cleanHyperparams(self):

        self.hyperparams[0] = clamp(abs(self.hyperparams[0]), 0.00000001 , 1)  # alpha,
        self.hyperparams[1] = clamp(abs(self.hyperparams[1]), 0.00000001 , 1)  # gamma,
        self.hyperparams[2] = clamp(abs(self.hyperparams[2]), 0.00000001 , 1)  # rho,
        self.hyperparams[3] = clamp(abs(self.hyperparams[3]), 0.00000001 , 1)  # epsilon,
        self.hyperparams[10] = clamp(abs(self.hyperparams[10]), 2, 100)  # hiddenLayerNeurons

    #checks all files that start with "id + 'result'",
    # e.g following files would be read if id = 0 ("0result", "0result_corners.txt", "0resultA.txt")
    def loadFitness(self):
        fitness = 0
        fileAmount = 0
        for file in os.listdir(dataPath):
            if file.startswith(str(self.id) + 'result'):
                fitness += calculateFitness(file)
                fileAmount += 1
        self.fitness = fitness/fileAmount



def clamp(toClamp, minValue, maxValue):
    return max(min(toClamp, maxValue), minValue)

def clamp01(toClamp):
    return clamp(toClamp, 0, 1)

def saveWeights(weights, filename):
    f = open(dataPath + filename, "w+")
    for j in range(len(weights)):

        f.write(str(weights[j][0]))
        for k in range(1, len(weights[0])):
            f.write("    " + str(weights[j][k]))
        f.write("\n")
    f.close()

def loadWeights(filename):
    with open(dataPath + filename) as f:
        return [[float(x) for x in line.split("    ")] for line in f]

def saveHyperparams(hyperParams, filename):
    f = open(dataPath + filename, "w+")
    f.write(str(abs(hyperParams[0])) + "\n")            # alpha,
    f.write(str(abs(hyperParams[1])) + "\n")            # gamma,
    f.write(str(abs(hyperParams[2])) + "\n")            # rho,
    f.write(str(abs(hyperParams[3])) + "\n")            # epsilon,
    f.write(str(hyperParams[4]) + "\n")                     # hitBulletReward,
    f.write(str(hyperParams[5]) + "\n")                     # hitByBulletReward,
    f.write(str(hyperParams[6]) + "\n")                     # hitRobotReward,
    f.write(str(hyperParams[7]) + "\n")                     # hitwallReward,
    f.write(str(hyperParams[8]) + "\n")                     # onDeathReward,
    f.write(str(hyperParams[9]) + "\n")                     # onWinReward,
    f.write(str(hyperParams[10]) + "\n")  # hiddenLayerNeurons
    f.close()

def loadHyperparams(filename):
    with open(dataPath + filename) as f:
        return [float(x) for x in f]

def avoidZero(min, max):
    rand = random.uniform(min, max)
    if rand == 0:
        rand = avoidZero(min, max)
        return rand
    else:
        return rand

def avoidZeroGauss(mean, standardeviation):
    rand = random.gauss(mean, standardeviation)
    if rand == 0:
        rand = avoidZeroGauss()
        return rand
    else:
        return rand
#
def generateWeights(inputNeurons, outputNeurons, weights_hidden=None, weights_output=None):
    # Generate random weights_hidden array with Normal distribution
    for i in range(populationSize):
        weights_hidden = [[random.gauss(0, randomWeightStandardDeviation) for k in range(inputNeurons + 1)] for j in range(int(robots[i].hyperparams[10]))]
        saveWeights(weights_hidden, str(i) + "weights_hidden.txt")

        weights_output = [[random.gauss(0, randomWeightStandardDeviation) for k in range(int(robots[i].hyperparams[10]) + 1)] for j in range(outputNeurons)]
        saveWeights(weights_output, str(i) + "weights_output.txt")

#genereates Hyperparameter, and saves them in robots and in files
def generateHyperparams():
    for i in range(populationSize):
        hyperParams = [0] * hyperparamAmount
        #hyperParams = [random.gauss(0, hyperparamStandardDeviation) for k in range(hyperparamAmount)]
        #scale to better fit 0 - 1
        for j in range(4):
            if j == 2:
                hyperParams[2] = avoidZero(0,1) * 0.01             #rho
            else:
                hyperParams[j] = avoidZero(0,1)                         # alpha,
        #hyperParams[1] = checkZero(0,1)                         # gamma,
        #hyperParams[2] = checkZero(0,1) * 0.001                 # rho,
        #hyperParams[3] = checkZero(0,1)                                            # epsilon,
        for j in range(4, 10):
            hyperParams[j] = avoidZeroGauss(0, 50)                         #hitBulletReward
        #yperParams[5] = checkZero(-500, 500)                               #hitByBulletReward
        #hyperParams[6] = checkZero(-500, 500)                            #hitRobotReward
        #hyperParams[7] = checkZero(-500, 500)                            #hitWallReward
        #hyperParams[8] = checkZero(-500, 500)                            #onDeathReward
        #hyperParams[9] = checkZero(-500, 500)                            #onWinReward
        hyperParams[10] = int(random.uniform(2, 20))                                # hiddenLayerNeurons,
        robots[i].hyperparams = hyperParams
        saveHyperparams(hyperParams, str(i) + "hyperparams.txt")



def loadAllHyperparams():
    hyperParams = [[0 for x in range(hyperparamAmount)] for y in range(populationSize)]
    for i in range(populationSize):
        hyperParams[i] = loadHyperparams(str(i) + "hyperparams.txt")
        robots[i].hyperparams = hyperParams[i]
    return hyperParams

def calculateFitness(filename):
    read_data = [str] * 2
    with open(dataPath + filename) as f:
        f.readline()
        f.readline()
        read_data[0] = f.readline()
        read_data[1] = f.readline()

    #ensure our robot is always on position 0
    if(read_data[0].find('ARL.Rl_nn') == -1):
        temp = read_data[0]
        read_data[0] = read_data[1]
        read_data[1] = temp

    #strip
    read_data[0] = read_data[0].split("\t")[1].split(' ')[0]
    read_data[1] = read_data[1].split("\t")[1].split(' ')[0]

    #fitness
    fitness = int(read_data[0])/(int(read_data[0]) + int(read_data[1]))
    return fitness

def runRoboCode(generations, battlePathAndName, resultPathandName):
    for x in range(generations):
        if (resultPathandName == ''):
            p1 = subprocess.Popen('java -Xmx512M -Dsun.io.useCanonCaches=false -DPARALLEL=true -cp libs/robocode.jar robocode.Robocode -battle ' + battlePathAndName + ' -tps 1000000 -nodisplay', cwd = robocodePath)
            subprocesses.append(p1)
        else:
            p1 = subprocess.Popen('java -Xmx512M -Dsun.io.useCanonCaches=false -DPARALLEL=true -cp libs/robocode.jar robocode.Robocode -battle ' + battlePathAndName + ' -results ' + resultPathandName + ' -tps 1000000 -nodisplay', cwd = robocodePath)
            subprocesses.append(p1)
        print("gen: " + str(x+1) + " of " + str(generations) + " done")


def selectParents(parents):
    out_amount = int(len(parents) * topParentPercent)
    parents.sort(key = lambda x: x.fitness, reverse = True)

    #Fill best_parents array with the best robots
    best_parents = []
    for i in range(out_amount):
        best_parents.append(parents[i])


    #Get robot with biggest diversity to best robot
    best_diversity_index = 1
    best_diversity = 0

    i = 1
    while i < len(best_parents):
        total_diversity = 0

        j = 0
        while j < len(parents[0].hyperparams):
            diversity = pow((best_parents[0].hyperparams[j] - best_parents[i].hyperparams[j])/abs(best_parents[0].hyperparams[j]), 2)

            total_diversity += diversity
            j += 1


        if total_diversity >=  best_diversity:

            best_diversity_index = i
            best_diversity = total_diversity

        i += 1

    #Put robot with biggest diversity from best robot to the first position of the output array
    output_array = [None] * out_amount
    output_array[0] = best_parents[best_diversity_index]

    best_parents_index = 0
    j = 1
    while j < out_amount:

        if j == (best_diversity_index + 1):
            best_parents_index += 1

        output_array[j] = best_parents[best_parents_index]
        best_parents_index += 1

        j += 1

    return output_array


def mutate(parent):
    child = deepcopy(parent)
    for i in range(len(child.hyperparams)):
        rand = random.random()
        if(rand < mutationChance):
            child.hyperparams[i] = avoidZeroGauss(child.hyperparams[i], math.log(abs(child.hyperparams[i]/2)) + 1, 2)
    if int(abs(child.hyperparams[len(child.hyperparams) - 1])) < 2:
        child.hyperparams[len(child.hyperparams) - 1] = int(abs(child.hyperparams[len(child.hyperparams) - 1])) + 2
    else:
        child.hyperparams[len(child.hyperparams) - 1] = int(abs(child.hyperparams[len(child.hyperparams) - 1]))
    return child

def crossover(father, mother):
    son = deepcopy(father)
    daughter = deepcopy(mother)
    for i in range(len(father.hyperparams)):
        rand = random.random()
        if(rand < 0.5):
            son.hyperparams[i] = father.hyperparams[i]
            daughter.hyperparams[i] = mother.hyperparams[i]
        else:
            son.hyperparams[i] = mother.hyperparams[i]
            daughter.hyperparams[i] = father.hyperparams[i]

    children = [son, daughter]
    print(father.hyperparams)
    print(mother.hyperparams)
    print(son.hyperparams)
    print(daughter.hyperparams)
    return children

def makeEvolution(parents):
    selectedParents = selectParents(parents)
    nextGen = [Robot] * populationSize
    nextGen[0] = selectedParents[0]
    nextGen[1] = selectedParents[1]

    crossover_number = populationSize - len(parents) + 2

    if crossover_number % 2 == 0:
        mutationNumber = len(selectedParents)
    else:
        mutationNumber = len(selectedParents) - 1

    i = 2

    while i < mutationNumber:
        nextGen[i] = mutate(selectedParents[i])
        i += 1

    while i < populationSize - 1:
        rand1 = int(random.random() * populationSize)
        rand2 = int(random.random() * populationSize)
        parent1 = parents[rand1]
        parent2 = parents[rand2]

        children = crossover(parent1, parent2)
        nextGen[i] = children[0]
        nextGen[i+1] = children[1]
        i += 2

    for i in range(populationSize):
        nextGen[i].id = i
    return nextGen

def changeEpsilonToOne(id):
    f = open(dataPath + str(id) + "hyperparams.txt" , "r")
    list_of_lines = f.readlines()
    list_of_lines[3] = "1\n"
    f = open(dataPath + str(id) + "hyperparams.txt", "w")
    f.writelines(list_of_lines)
    f.close()

def resetConfig():
    f = open(dataPath + "config.txt","w+")
    f.write("0\n")
    f.close()

def setConfig(id):
    f = open(dataPath + "config.txt","w+")
    f.write( str(id) +  "\n")
    f.close()

def init(populationSize):
    generateHyperparams()
    generateWeights(inputNeurons,outputNeurons)
    resetConfig()

robots = [Robot] * populationSize
for i in range(0,populationSize):
    robots[i] = Robot(i)


def saveFitness(filename):
    f = open(dataPath + filename, "a")
    f.write( str(robots[1].fitness) +  "\n")
    f.close()

def run(generations):
    global robots
    for r in robots:
        r.loadHyperparams()

    for i in range(generations):

        resetConfig()
        subprocesses.clear()
        for j in range(0,populationSize):
            #training stage
            runRoboCode(1,robocodePath + 'battles/NNTrain.battle', '')
            time.sleep(2)

        for s in subprocesses:
            s.wait()

        subprocesses.clear()

        resetConfig()
        #evaluation stage
        for j in range(0, populationSize):
            changeEpsilonToOne(j)
            runRoboCode(1,robocodePath + 'battles/NNEvaluate.battle', dataPath + str(j) + "result.txt")
            time.sleep(2)

        for s in subprocesses:
            s.wait()

        for r in robots:
            r.loadFitness()

        robots = makeEvolution(robots)
        for r in robots:
            r.cleanHyperparams()
            r.saveHyperparams()

        generateWeights(inputNeurons, outputNeurons)
        resetConfig()
        saveFitness("generationInfo.txt")

run(20)
#init(populationSize)
