import random
import subprocess
import os, sys
robocodePath = 'D:/FHTech/Sem2/ARL/Robocode/INSTALL/'
dataPath = os.path.dirname(sys.argv[0]) + '/out/production/RobocodeDeepQLearning/ARL/Rl_nn.data/'
battlePath = 'battles/' #path startign at the robocode directory
battleFileName = 'NNTrain.battle'
resultFileName = "result.txt"

inputNeurons = 6
hiddenLayerNeurons = 10
outputNeurons = 6
populationSize = 2
randomWeightStandardDeviation = 5
hyperparamAmount = 10
hyperparamStandardDeviation = 10
# hyperparams:
# alpha,
# gamma,
# rho,
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
    def __init__(self, id):
        self.id = id

    def loadHyperparams(self):
        self.hyperparams = loadHyperparams( str(self.id) + 'hyperparams.txt')

    def saveHyperparams(self):
        saveHyperparams(self.hyperparams, str(self.id) + 'hyperparams.txt')

    def loadFitness(self):
        self.fitness = calculateFitness(str(self.id) + resultFileName)

def clamp(toClamp, minValue, maxValue):
    return max(min(toClamp, maxValue), minValue)

def clamp01(toClamp):
    return clamp(toClamp, 0, 1)

def saveWeights(weights, filename):
    f = open(dataPath + filename, "w")
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
    f = open(dataPath + filename, "w")
    f.write(str(clamp01(hyperParams[0])) + "\n")            # alpha,
    f.write(str(clamp01(hyperParams[1])) + "\n")            # gamma,
    f.write(str(clamp01(hyperParams[2])) + "\n")            # rho,
    f.write(str(hyperParams[3]) + "\n")                     # hitBulletReward,
    f.write(str(hyperParams[4]) + "\n")                     # hitByBulletReward,
    f.write(str(hyperParams[5]) + "\n")                     # hitRobotReward,
    f.write(str(hyperParams[6]) + "\n")                     # hitwallReward,
    f.write(str(hyperParams[7]) + "\n")                     # onDeathReward,
    f.write(str(hyperParams[8]) + "\n")                     # onWinReward,
    f.write(str(int(clamp(hyperParams[9], 5, 30))) + "\n")  # hiddenLayerNeurons
    f.close()

def loadHyperparams(filename):
    with open(dataPath + filename) as f:
        return [float(x) for x in f]

def generateWeights(inputNeurons, hiddenLayerNeurons, outputNeurons, weights_hidden=None, weights_output=None):
    # Generate random weights_hidden array with Normal distribution
    for i in range(populationSize):
        weights_hidden = [[random.gauss(0, randomWeightStandardDeviation) for k in range(inputNeurons + 1)] for j in range(hiddenLayerNeurons)]
        saveWeights(weights_hidden, str(i) + "weights_hidden.txt")

        weights_output = [[random.gauss(0, randomWeightStandardDeviation) for k in range(hiddenLayerNeurons + 1)] for j in range(outputNeurons)]
        saveWeights(weights_output, str(i) + "weights_output.txt")

def generateHyperparams():
    hyperParams = [populationSize]
    for i in range(populationSize):
        hyperParams = [random.gauss(0, hyperparamStandardDeviation) for k in range(hyperparamAmount)]
        #scale to better fit 0 - 1
        hyperParams[0] = (hyperParams[0] / hyperparamStandardDeviation) * 0.3 + 0.5         # alpha,
        hyperParams[1] = (hyperParams[1] / hyperparamStandardDeviation) * 0.3 + 0.5         # gamma,
        hyperParams[2] = (hyperParams[2] / hyperparamStandardDeviation) * 0.00005 + 0.0001  # rho,
        hyperParams[9] =  hyperParams[9] + 10                                               # hiddenLayerNeurons,
        saveHyperparams(hyperParams, str(i) + "hyperparams.txt")

def loadAllHyperparams():
    hyperParams = [[0 for x in range(hyperparamAmount)] for y in range(populationSize)]
    for i in range(populationSize):
        hyperParams[i] = loadHyperparams(str(i) + "hyperparams.txt")
    return hyperParams

def calculateFitness(filename):
    read_data = ['lol'] * 2
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
        p1 = subprocess.Popen('java -Xmx512M -Dsun.io.useCanonCaches=false -DPARALLEL=true -cp libs/robocode.jar robocode.Robocode -battle ./' + battlePathAndName + ' -results ' + resultPathandName + ' -tps 1000000 -nodisplay', cwd = robocodePath)
        p1.wait()
        print("gen: " + str(x+1) + " of " + str(generations) + " done")

r = Robot(0)
#runRoboCode(1,battlePath + battleFileName, dataPath + str(r.id) + resultFileName)
r.loadFitness()
r.loadHyperparams()