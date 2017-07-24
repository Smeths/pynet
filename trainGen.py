#An example of a class
import numpy
import scipy.special
import neuralNetworkGen as net

# number of inputs, hidden and output
input_nodes = 784
hidden_nodes = [100]
output_nodes = 10

# learning rate is 0.3
learning_rate = 0.3

# create instance of neural network
n = net.neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
#print "layers = ",n.nlayers
#print "nodes = ",n.nodes
#print "weights = ",n.w[0].shape,n.w[1].shape

# load the mnist training data
training_data_file = open("data/mnist_train.csv",'r')
training_data_list = training_data_file.read().splitlines()
training_data_file.close()

# train the neural network

# go through all records in thc
for record in training_data_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    # scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:])/255.0)*0.99 + 0.01
    # create the target output values (all 0.01 except the desired label which is 0.99)
    targets = numpy.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs,targets)

# writing the weights to a file

w0_data_file = open("data/w0.csv",'w')
numpy.savetxt(w0_data_file,n.w[0])
w0_data_file.close()

w1_data_file = open("data/w1.csv",'w')
numpy.savetxt(w1_data_file,n.w[1])
w1_data_file.close()


