#An example of a class
import numpy
import scipy.special
import neuralNetwork2 as net

# number of inputs, hidden and output
input_nodes = 784
hidden1_nodes = 100
hidden2_nodes = 50
output_nodes = 10

# learning rate is 0.3
learning_rate = 0.3

# create instance of neural network
n = net.neuralNetwork(input_nodes,hidden1_nodes,hidden2_nodes,output_nodes,learning_rate)

# load the mnist training data
training_data_file = open("data/mnist/mnist_train.csv",'r')
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

wih1_data_file = open("data/weights/wih1.csv",'w')
numpy.savetxt(wih1_data_file,n.wih1)
wih1_data_file.close()

wh1h2_data_file = open("data/weights/wh1h2.csv",'w')
numpy.savetxt(wh1h2_data_file,n.wh1h2)
wh1h2_data_file.close()

wh2o_data_file = open("data/weights/wh2o.csv",'w')
numpy.savetxt(wh2o_data_file,n.wh2o)
wh2o_data_file.close()

