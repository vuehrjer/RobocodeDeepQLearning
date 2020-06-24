import random
import subprocess
import os, sys
import numpy as np
from copy import deepcopy
import fileinput
robocodePath = 'D:/FHTech/Sem2/ARL/Robocode/INSTALL/'
dataPath = os.path.dirname(sys.argv[0]) + '/out/production/RobocodeDeepQLearning/ARL/Rl_nn.data/'
battleFileName = 'battles/NNTrain.battle'


topParentPercent = 0.8
mutationChance = 0.8
inputNeurons = 7
hiddenLayerNeurons = 10
outputNeurons = 1
populationSize = 10
randomWeightStandardDeviation = 5
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

        self.hyperparams[0] = clamp01(self.hyperparams[0])  # alpha,
        self.hyperparams[1] = clamp01(self.hyperparams[1])  # gamma,
        self.hyperparams[2] = clamp01(self.hyperparams[2])  # rho,
        self.hyperparams[3] = clamp01(self.hyperparams[3])  # epsilon,
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
    f.write(str(clamp01(hyperParams[0])) + "\n")            # alpha,
    f.write(str(clamp01(hyperParams[1])) + "\n")            # gamma,
    f.write(str(clamp01(hyperParams[2])) + "\n")            # rho,
    f.write(str(clamp01(hyperParams[3])) + "\n")            # epsilon,
    f.write(str(hyperParams[4]) + "\n")                     # hitBulletReward,
    f.write(str(hyperParams[5]) + "\n")                     # hitByBulletReward,
    f.write(str(hyperParams[6]) + "\n")                     # hitRobotReward,
    f.write(str(hyperParams[7]) + "\n")                     # hitwallReward,
    f.write(str(hyperParams[8]) + "\n")                     # onDeathReward,
    f.write(str(hyperParams[9]) + "\n")                     # onWinReward,
    f.write(str(int(clamp(abs(hyperParams[10]), 2, 100))) + "\n")  # hiddenLayerNeurons
    f.close()

def loadHyperparams(filename):
    with open(dataPath + filename) as f:
        return [float(x) for x in f]

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
    hyperParams = [hyperparamAmount]
    for i in range(populationSize):
        hyperParams = [random.gauss(0, hyperparamStandardDeviation) for k in range(hyperparamAmount)]
        #scale to better fit 0 - 1
        hyperParams[0] = clamp01(abs(hyperParams[0] / hyperparamStandardDeviation) * 0.3 + 0.5)         # alpha,
        hyperParams[1] = clamp01(abs(hyperParams[1] / hyperparamStandardDeviation) * 0.3 + 0.5)         # gamma,
        hyperParams[2] = clamp01(abs(hyperParams[2] / hyperparamStandardDeviation) * 0.00005 + 0.0001)  # rho,
        hyperParams[3] = random.random()                                                                # epsilon,
        hyperParams[10] = int(clamp(abs(hyperParams[10] + 10), 2, 100))                                 # hiddenLayerNeurons,
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
            p1.wait()
        else:
            p1 = subprocess.Popen('java -Xmx512M -Dsun.io.useCanonCaches=false -DPARALLEL=true -cp libs/robocode.jar robocode.Robocode -battle ' + battlePathAndName + ' -results ' + resultPathandName + ' -tps 1000000 -nodisplay', cwd = robocodePath)
            p1.wait()
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
            diversity = pow((best_parents[0].hyperparams[j] - best_parents[i].hyperparams[j]), 2)

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
            child.hyperparams[i] = np.random.normal(loc=child.hyperparams[i], scale=abs(child.hyperparams[i]/2))
    child.hyperparams[len(child.hyperparams) - 1] = int(child.hyperparams[len(child.hyperparams) - 1])
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


def run(generations):
    global robots
    for r in robots:
        r.loadHyperparams()

    for i in range(generations):
        for j in range(0,populationSize):
            #training stage
            runRoboCode(1,robocodePath + battleFileName, '')
            changeEpsilonToOne(j)
            setConfig(j)

            #evaluation stage
            runRoboCode(1,robocodePath + battleFileName, dataPath + str(j) + "result.txt")


        for r in robots:
            r.loadFitness()

        robots = makeEvolution(robots)
        for r in robots:
            r.cleanHyperparams()
            r.saveHyperparams()
        generateWeights(inputNeurons, outputNeurons)
        resetConfig()


run(3)
#init(populationSize)