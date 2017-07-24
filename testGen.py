#An example of a class
import numpy
import scipy.special
import neuralNetworkGen as net

# number of inputs, hidden and output
input_nodes = 784
hidden_nodes = 100
output_nodes = 10

# learning rate is 0.3
learning_rate = 0.3

# create instance of neural network
n = net.neuralNetwork(input_nodes,[hidden_nodes],output_nodes,learning_rate)

# overwriting the weights used in the network
w0_data_file = open("data/weights/w0.csv",'r')
w0_data = w0_data_file.read().split()
w0_data = [float(i) for i in w0_data]
n.w = [numpy.array(w0_data).reshape(hidden_nodes, input_nodes)]
w0_data_file.close()

# overwriting the weights used in the network
w1_data_file = open("data/weights/w1.csv",'r')
w1_data = w1_data_file.read().split()
w1_data = [float(i) for i in w1_data]
n.w.append(numpy.array(w1_data).reshape(output_nodes, hidden_nodes))
w1_data_file.close()

# creating a diagnostic file
test_diag_file = open("data/peformance/testGen_diagnostics.csv",'w')
test_err_records = open("data/performance/testGen_err_records.csv",'w')

# checking against the test set
test_data_file = open("data/mnist/mnist_test.csv",'r')
test_data = test_data_file.read().splitlines()
test_data_file.close()
score = 0
fail0 = [0,0,0,0,0,0,0,0,0,0,0]
fail1 = [0,0,0,0,0,0,0,0,0,0,0]
fail2 = [0,0,0,0,0,0,0,0,0,0,0]
fail3 = [0,0,0,0,0,0,0,0,0,0,0]
fail4 = [0,0,0,0,0,0,0,0,0,0,0]
fail5 = [0,0,0,0,0,0,0,0,0,0,0]
fail6 = [0,0,0,0,0,0,0,0,0,0,0]
fail7 = [0,0,0,0,0,0,0,0,0,0,0]
fail8 = [0,0,0,0,0,0,0,0,0,0,0]
fail9 = [0,0,0,0,0,0,0,0,0,0,0]

# go through all records in thc
for record in test_data:
    # split the record by the ',' commas
    all_values = record.split(',')
    # scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:])/255.0)*0.99 + 0.01
    input_num = int(all_values[0])
    out = n.query(inputs)
    max_tup = numpy.where(out==out.max())
    output_num = [x[0] for x in max_tup]
    output_num = numpy.asscalar(output_num[0])
    if input_num == output_num:
        score = score + 1
    else:
        test_err_records.write(record + '\n')
        if input_num == 0:
            fail0[10] = fail0[10] + 1
            fail0[output_num] = fail0[output_num] + 1 
        elif input_num == 1:
            fail1[10] = fail1[10] + 1
            fail1[output_num] = fail1[output_num] + 1 
        elif input_num == 2:
            fail2[10] = fail2[10] + 1
            fail2[output_num] = fail2[output_num] + 1 
        elif input_num == 3:
            fail3[10] = fail3[10] + 1
            fail3[output_num] = fail3[output_num] + 1 
        elif input_num == 4:
            fail4[10] = fail4[10] + 1
            fail4[output_num] = fail4[output_num] + 1
        elif input_num == 5:
            fail5[10] = fail5[10] + 1
            fail5[output_num] = fail5[output_num] + 1
        elif input_num == 6:
            fail6[10] = fail6[10] + 1
            fail6[output_num] = fail6[output_num] + 1
        elif input_num == 7:
            fail7[10] = fail7[10] + 1
            fail7[output_num] = fail7[output_num] + 1
        elif input_num == 8:
            fail8[10] = fail8[10] + 1
            fail8[output_num] = fail8[output_num] + 1
        elif input_num == 9:
            fail9[10] = fail9[10] + 1
            fail9[output_num] = fail9[output_num] + 1

success_percent = str(100*float(score)/10000)

test_diag_file.write("number of 0s failed: " + str(fail0[10]) + "\n")
test_diag_file.write(str(fail0[0:10]) + "\n")
test_diag_file.write("number of 1s failed: " + str(fail1[10]) + "\n")
test_diag_file.write(str(fail1[0:10]) + "\n")
test_diag_file.write("number of 2s failed: " + str(fail2[10]) + "\n")
test_diag_file.write(str(fail2[0:10]) + "\n")
test_diag_file.write("number of 3s failed: " + str(fail3[10]) + "\n")
test_diag_file.write(str(fail3[0:10]) + "\n")
test_diag_file.write("number of 4s failed: " + str(fail4[10]) + "\n")
test_diag_file.write(str(fail4[0:10]) + "\n")
test_diag_file.write("number of 5s failed: " + str(fail5[10]) + "\n")
test_diag_file.write(str(fail5[0:10]) + "\n")
test_diag_file.write("number of 6s failed: " + str(fail6[10]) + "\n")
test_diag_file.write(str(fail6[0:10]) + "\n")
test_diag_file.write("number of 7s failed: " + str(fail7[10]) + "\n")
test_diag_file.write(str(fail7[0:10]) + "\n")
test_diag_file.write("number of 8s failed: " + str(fail8[10]) + "\n")
test_diag_file.write(str(fail8[0:10]) + "\n")
test_diag_file.write("number of 9s failed: " + str(fail9[10]) + "\n")
test_diag_file.write(str(fail9[0:10]) + "\n")
test_diag_file.write("proportion passed: " + success_percent + "%" + "\n")

test_diag_file.close()
test_err_records.close()
    
