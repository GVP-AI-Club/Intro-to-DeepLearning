import numpy as np
X = np.array(([0,0,1],[0,1,1],[1,0,1],[1,1,1]),dtype=float)
y = np.array(([0],[1],[1],[0]),dtype=float)

def sigmoid(x):
    return (1/(1+np.exp(-x)))


def sigmoid_derivative(p):
    return(p*(1-p))


class NeuralNetwork:
    def __init__(self,x,y):
        self.input = x
        #the 4 in the below line means that we
        # will have 4 hidden nodes/4 nodes in the hidden layer
        self.weights1 = np.random.rand(self.input.shape[1],4) 
        self.weights2 = np.random.rand(4,1)
        self.y = y
        self.output = np.zeros(y.shape)


    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input,self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1,self.weights2))
        return (self.layer2)

    def backprop(self):
        d_weights2 = np.dot(self.layer1.T,2*(self.y-self.output)*sigmoid_derivative(self.output))
        d_weights1 = np.dot(self.input.T,np.dot(2*(self.y-self.output)*sigmoid_derivative(self.output),self.weights2.T)*sigmoid_derivative(self.layer1))

        self.weights1+=d_weights1
        self.weights2+=d_weights2

    def train(self,X,y):
        self.output = self.feedforward()
        self.backprop()


NN = NeuralNetwork(X,y)
for i in range(1000): #this network is trained n times
    if i%100 == 0:
        print("for iteration #"+ str(i)+'\n')
        print("Input:\n",str(X))
        print("Actual Output:\n" + str(y))
        print("Predicted Output:\n" + str(NN.feedforward()))
        print("Loss:\n" + str(np.mean(np.square(y-NN.feedforward()))))
        print("\n")
    
    NN.train(X,y)


