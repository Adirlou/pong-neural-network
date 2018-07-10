from pong_ui import *
from pong_state import *
from pong_neural_network import *

#Read the expert policy
data = readData("expert_policy.txt")

#Compute minibatch gradient descent
"""ws, bs, lossArray, accuracyArray = minibatchGD(data, epoch=800, batchSize = 128, d=5, dPrime=128, weightScale=0.04)

#Store the weight, matrices, bias, loss and accuracy on the local computer
for index, weight in enumerate(ws):
    np.save("W"+str(index)+".npy", weight)

for index, bias in enumerate(bs):
    np.save("B"+str(index)+".npy", bias)

with open('lossArray.txt', 'w') as fileout:
    for item in lossArray:
        fileout.write("%s\n" % item)

with open('accuracyArray.txt', 'w') as fileout2:
    for item in accuracyArray:
        fileout2.write("%s\n" % item)"""

#Get matrices and bias for the deep network
W1 = np.load("W1.npy")
W2 = np.load("W2.npy")
W3 = np.load("W3.npy")
W4 = np.load("W4.npy")

b1 = np.load("B1.npy")
b2 = np.load("B2.npy")
b3 = np.load("B3.npy")
b4 = np.load("B4.npy")

Ws = [W1, W2, W3, W4]
bs = [b1, b2, b3, b4]

#Set the initial state
currentState = State.initialState()
currentScore = 0

while True:
    #If we lose, just start another game
    if currentState.gameOver():
        currentState = State.initialState()
        currentScore = 0

    #Choose best action depending on output of the deep network
    action = Action(bestAction(currentState.returnListofAttributes(), Ws, bs))

    #Update state
    currentState = currentState.nextState(action)

    #If we are able to rebounce the ball, increase score
    if currentState.reward == 1:
        currentScore += 1

    #Display the state with PyGame
    draw(window, currentState, currentScore)
    pygame.display.update()
    fps.tick(30)
