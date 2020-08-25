import numpy as np

inputsArr = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])  #inputs of function
outputArr = np.array([-1,-1,-1,1])  #outputs of fucntion

w = np.random.random((2,1))  #init weights
#print(w)

bios = 0.1
e = .1 #eta

def gradiantD(w , inputsArr , outputArr,e ,bios ):
    for i in range(1000):
        for j in range(len(inputsArr)):
            sum = w[0]*inputsArr[j][0] + w[1]*inputsArr[j][1] +bios
            err = outputArr[j] - sum
            bios = bios + err*e
            w[1]=  w[1] + e * inputsArr[j][1] *err
            w[0] = w[0] + e * inputsArr[j][0] * err
    lastw2=w[1]
    lastw1=w[0]
    lastbios = bios
    print( "weight of first input :" , lastw1 )
    print("weight of second input :", lastw2)
    print("last amount of bios :", bios)
    return  lastw2 ,lastw1 , lastbios


lastw2 ,lastw1 ,lastbios = gradiantD(w , inputsArr ,outputArr , e , bios)



for input in inputsArr:
    sum = input[0]*lastw1 + input[1]*lastw2 + lastbios
    if sum >= 0:
        print(" %d , %d, output %d" % (input[0],input[1], 1))
    else:
        print(" %d , %d, output %d" % (input[0], input[1], -1))



