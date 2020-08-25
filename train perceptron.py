
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("p1-dataset.txt", delimiter=",") #read from file

max = 100 #for normalizing
min = 0
for i in range(data.shape[0]): #normalizing data
	data[i][0] = data[i][0] / max-min
	data[i][1] = data[i][1] / max-min


w = np.random.random((2,1))  #init weights

khata = np.zeros(500)    #for errors

bios = 0.1
e = .1 #eta





def gradiantD(w, bios, e, data):

	for i in range(500):
		eror = 0
		for j in range(len(data)):
			sum = w[0] * data[j][0] + w[1] * data[j][1] + bios
			if data[j][2] == 0:
				data[j][2] = -1
			err = data[j][2] - sum

			w[1] = w[1] + e * data[j][1] * err #update weights
			w[0] = w[0] + e * data[j][0] *err
			bios = bios + e * err #update bios
			eror = eror + (err**2)/2
		khata[i] = eror
	return   khata,bios, w[0], w[1]




khata, bios, w1, w2= gradiantD(w, bios, e , data)



printx= data[:,:-1]
printy = data[:,-1]
Z = (-bios - (w1 * printx)) / w2
shape = plt.figure()
plt.plot(np.arange(0, 500), khata)
plt.show()
plt.figure()
plt.scatter(printx[:, 0], printx[:, 1], marker="x", c=printy)
plt.plot(printx, Z, "-")
plt.show()
