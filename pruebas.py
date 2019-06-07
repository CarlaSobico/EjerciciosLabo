from numpy import *
data=array([[1,2,3,4,5],
      [6,7,8,9,10],
      [11,12,13,14,15]])
data1=array([[1,1,1,1,1],
            [1,1,1,1,1],[1,1,1,1,1]])
data1[0]=data[1]
data[1]=data[2]
data[2]=data[0]
data1[1]=[7,7,7,7,7]
print(data)
print(data1)