robocodePath = 'D:/FHTech/Sem2/ARL/Robocode/INSTALL/'
dataPath = 'D:/FHTech/Sem2/ARL/Repo/RobocodeDeepQLearning/out/production/RobocodeDeepQLearning/ARL/Rl_nn.data/'
inputNeurons = 6
hiddenLayerNeurons = 10
outputNeurons = 6
populationSize = 1
# generations =
randomWeightStandardDeviation = 5

def saveWeights(weights, filename):
    f = open(dataPath + filename, "w")
    for j in range(len(weights)):
        for k in range(len(weights[0])):
            f.write(str(weights[j][k]) + "    ")
        f.write("\n")
    f.close()


import random


def generateWeights(inputNeurons, hiddenLayerNeurons, outputNeurons, weights_hidden=None, weights_output=None):
    # Generate random weights_hidden array with Normal distribution
    for i in range(populationSize):
        weights_hidden = [[random.gauss(0, randomWeightStandardDeviation) for k in range(inputNeurons + 1)] for j in range(hiddenLayerNeurons)]
        saveWeights(weights_hidden, str(i) + "weights_hidden.txt")

        weights_output = [[random.gauss(0, randomWeightStandardDeviation) for k in range(hiddenLayerNeurons + 1)] for j in range(outputNeurons)]
        saveWeights(weights_output, str(i) + "weights_output.txt")


generateWeights(inputNeurons, hiddenLayerNeurons, outputNeurons)
