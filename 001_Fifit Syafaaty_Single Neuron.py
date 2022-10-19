#Fifit Syafaaty
#21091397001
#2021 A
#single neuron (Pakai numpy)

#setarakan numpy
import numpy as np

#setarakan variabel dengan jumlah input 10
inputs = [4, 7, 8, 2, 1, 9, 3, 6, 5, 10]

#panjang weights sesuai dengan panjang input, dan jumlah weights harus sama dengan jumlah neuron
weights = [0.4, 0.1, 0.7, 0.9, 0.11, 0.8, 0.3, 0.5, 0.10, -0.7]

#jumlah bias harus sama dengan jumlah neuron
bias = 7

#output
output = np.dot(weights, inputs) + bias

#print output
print(output)