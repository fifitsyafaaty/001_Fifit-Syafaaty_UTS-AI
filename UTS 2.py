#Fifit Syafaaty
#21091397001
#2021 A
#Multi Neuron Batch Input

#setarakan numpy
import numpy as np

#setarakan variabel dengan matriks 6x10 (input 10 dan batch 6)
inputs = [[0.4, 6.0, 0.7, 9.0, 0.1, 7.0, 0.2, 5.0, 0.9, 1.0],
          [0.7, 8.0, 0.5, 0.3, 9.0, 1.0, 5.0, 1.0, 5.0, 5.0],
          [4.8, 1.1, 8.0, 9.0, 3.7, 3.9, 4.3, 7.4, 3.0, 1.2],
          [2.3, 3.9, 4.5, 0.6, 7.7, 4.5, 0.2, 2.6, 8.8, 5.0],
          [2.2, 4.9, 0.9, 7.0, 1.0, 1.1, 5.0, 4.2, 8.0, 1.4],
          [1.0, 9.3, 8.0, 4.9, 2.1, 3.7, 0.1, 7.0, 2.0, 3.0]]

#panjang weights sesuai dengan panjang input, dan jumlah weights harus sama  dengan jumlah neuron
weights1 = [[0.4, 6.2, 4.0, 8.1, 3.6, 7.0, 5.8, 2.5, 2.1, 1.0],
           [4.5, 9.8, 0.4, 3.0, 2.0, 6.0, 3.7, 8.1, 0.2, 3.4],
           [1.7, 7.9, 2.0, 0.9, 5.9, 9.7, 2.2, 4.6, 1.4, 3.8],
           [0.1, 5.0, 1.5, 7.8, 5.4, 3.7, 3.0, 4.4, 7.9, 4.0],
           [1.0, 1.5, 7.2, 3.7, 3.3, 9.9, 2.3, 0.1, 8.8, 3.3]]

#jumlah bias harus sama dengan jumlah neuron
biases1 = [9.0, 4.0, 5.5, 7.0, 2.2]

#panjang weight sesuai dengan panjang input, dan jumlah weight harus sama dengan jumlah neuron
weights2 = [[2.0, 3.5, 6.0, 1.0, 3.3],
            [2.3, 4.3, 0.1, 2.8, 2.1],
            [5.7, 1.5, 2.7, 9.2, 1.5]]

#jumlah bias harus sesuai dengan jumlah neuron
biases2 =[1.3, 4.0, 2.2]


#ouputs
layer1_outputs = np.dot(inputs, np.array(weights1).T) + biases1
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

#print ouputs
print(layer2_outputs)