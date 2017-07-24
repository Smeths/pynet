#An example of a class
import numpy
import scipy.special

class neuralNetwork:

    def __init__(self, inputnodes, hiddennodes1, hiddennodes2, outputnodes, learningrate):
        # node information
        self.inodes = inputnodes
        self.hnodes1 = hiddennodes1
        self.hnodes2 = hiddennodes2
        self.onodes = outputnodes
        self.lr = learningrate
        self.wih1 = numpy.random.normal(0.0, pow(self.hnodes1, -0.5), (self.hnodes1, self.inodes))
        self.wh1h2 = numpy.random.normal(0.0, pow(self.hnodes2, -0.5), (self.hnodes2, self.hnodes1))
        self.wh2o = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes2))
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list,targets_list):

        # converts inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list,ndmin=2).T
        
        # calculate signals into hidden1 layer
        hidden1_inputs = numpy.dot(self.wih1, inputs)
        # calculate the signals emerging from the hidden layer
        hidden1_outputs = self.activation_function(hidden1_inputs)

        # calculate signals into hidden2 layer
        hidden2_inputs = numpy.dot(self.wh1h2, hidden1_outputs)
        # calculate the signals emerging from the hidden layer
        hidden2_outputs = self.activation_function(hidden2_inputs)

        # calculate the signals into the final output layer
        final_inputs = numpy.dot(self.wh2o,hidden2_outputs)        
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden2 layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden2_errors = numpy.dot(self.wh2o.T, output_errors)
        # hidden1 layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden1_errors = numpy.dot(self.wh1h2.T, hidden2_errors)

        # update the weights for the links between the hidden and output layers
        self.wh2o += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), 
                                          numpy.transpose(hidden2_outputs))
        # update the weights for the links between the input and the hidden layers
        self.wh1h2 += self.lr * numpy.dot((hidden2_errors * hidden2_outputs * (1.0 - hidden2_outputs)),
                                           numpy.transpose(hidden1_outputs))
        # update the weights for the links between the input and the hidden layers
        self.wih1 += self.lr * numpy.dot((hidden1_errors * hidden1_outputs * (1.0 - hidden1_outputs)),
                                          numpy.transpose(inputs))

    def query(self, inputs_list):
        # converts inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden1_inputs = numpy.dot(self.wih1, inputs)
        # calculate the signals emerging from the hidden layer
        hidden1_outputs = self.activation_function(hidden1_inputs)

        # calculate signals into hidden layer
        hidden2_inputs = numpy.dot(self.wh1h2, hidden1_outputs)
        # calculate the signals emerging from the hidden layer
        hidden2_outputs = self.activation_function(hidden2_inputs)

        # calculate the signals into the final output layer
        final_inputs = numpy.dot(self.wh2o,hidden2_outputs)        
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs










