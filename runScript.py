robocodePath = 'D:/FHTech/Sem2/ARL/Robocode/INSTALL/'
dataPath = 'D:/FHTech/Sem2/ARL/Repo/RobocodeDeepQLearning/out/production/RobocodeDeepQLearning/ARL/Rl_nn.data/'
battlePath = './battles/NNTrainCrazy.battle' #path startign at the robocode directory
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
    def __init__(self, id, fitness):
        self.id = id
        self.fitness = fitness

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
    with open(filename) as f:
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

import random
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

import subprocess
def runRoboCode(generations, battlepath):
    for x in range(generations):
        p1 = subprocess.Popen('java -Xmx512M -Dsun.io.useCanonCaches=false -DPARALLEL=true -cp libs/robocode.jar robocode.Robocode -battle ' + battlepath + ' -tps 1000000 -nodisplay', cwd = robocodePath)
        p1.wait()
        print("gen: " + str(x+1) + " of " + str(generations) + " done")


