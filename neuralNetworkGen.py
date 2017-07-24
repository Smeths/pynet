#An example of a class
import numpy
import scipy.special

class neuralNetwork:

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # node information
        nodes = [inputnodes] + hiddennodes + [outputnodes]
        nlayers = len(nodes)
        self.nlayers = nlayers
        self.nodes = nodes
        self.lr = learningrate
        weights = []
        for i in range(0,nlayers-1):
            weight = numpy.random.normal(0.0, pow(nodes[i+1], -0.5) , (nodes[i+1], nodes[i]))
            weights.append(weight)
        self.w = weights
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list,targets_list):

        # converts inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list,ndmin=2).T
        
        # forward propagation
        layers = [inputs]

        for i in range(0,self.nlayers-1):
            layer = numpy.dot(self.w[i], layers[i])
            layer = self.activation_function(layer)
            layers.append(layer)

        # backward propagation
        errors = [targets - layers[self.nlayers-1]]
        
        count = 0
        for i in range(self.nlayers-2,-1,-1):
            self.w[i] += self.lr * numpy.dot((errors[count] * layers[i+1] * (1.0 - layers[i+1])), 
                                              numpy.transpose(layers[i]))
            error = numpy.dot(self.w[i].T, errors[count])
            errors.append(error)
            count = count + 1

    def query(self, inputs_list):
        # converts inputs list to 2d array
        layer = numpy.array(inputs_list, ndmin=2).T
        for i in range(0,self.nlayers-1):
            layer = numpy.dot(self.w[i], layer)
            layer = self.activation_function(layer)
        return layer
        
        

        
