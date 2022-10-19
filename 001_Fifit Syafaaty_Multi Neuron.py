#Fifit Syafaaty
#21091397001
#2021 A
#Multi neuron (Pakai numpy)

#setarakan numpy
import numpy as np

#setarakan variabel dengan jumlah input 10
inputs = [4.0, 2.0, 7.0, 6.0, 1.0, 12.0, 9.0, 13.0, 17.0, 11.0]

#panjang weights sesuai dengan panjang input, dan jumlah weights harus sama dengan jumlah neuron
weights = [[0.6, 0.9, 0.7, 0.3, 0.1, 0.3, 0.5, 0.4, 0.11, -0.9],
           [0.12, 0.22, 0.8, 0.1, 0.17, 0.23, 0.21, 0.95, 0.99, 0.85],
           [0.25, 0.19, 0.21, 0.42, 0.33, -0.34, -0.56, -0.22, 0.12, 0.15],
           [-4.0, 0.44, 0.29, -0.30, 0.62, 9.0, 1.0, 0.55, -0.47, 0.12],
           [1.0, 0.8, -0.3, 7.0, -0.78, -0.38, 10.0, -0.8, 0.1, -0.11]]

#jumlah bias harus sama dengan jumlah neuron
biases = [4.0, 7.0, 17.0, 13.0, 1.0]

#output
layer_outputs = np.dot(weights, inputs) + biases

#print output
print(layer_outputs)