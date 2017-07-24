import numpy
import matplotlib.pyplot as plt

data_file = open("data/performance/test_err_records.csv",'r')
data_list = data_file.readlines()
data_file.close()
count = 1

for record in data_list:
    all_values = record.split(',')
    image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
    plt.imshow(image_array, cmap='Greys', interpolation='None')
    image = "plots/digit_"+str(all_values[0]) + "_" + str(count)
    plt.savefig(image)
    count = count + 1
