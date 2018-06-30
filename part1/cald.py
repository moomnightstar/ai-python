from scipy import sparse
import numpy as np
from sklearn import preprocessing
import linecache
import random

count=0
n=1000
row = []
col = []
data = []
idf= np.loadtxt(open("idf.txt","r"),delimiter=",")
for i in range(n):
    a = random.randrange(0, 50000)
    line = linecache.getline('data', a)
    line = line.strip().split(' ')
    comment = line[1:]
    for i in range(len(comment)):
        index_value = comment[i].split(":")
        index = int(index_value[0])
        value = int(index_value[1])
        col.append(index)
        data.append(value * idf[index])
        row.append(count)
    count = count + 1
testdata = sparse.coo_matrix((data, (row, col)), shape=(n, 89527))
testdata=preprocessing.normalize(testdata, norm='l1')
testdata=np.array(testdata.todense())
result=0
for i in range(n):
    for j in range(n):
        if i!=j:
            result += np.dot((testdata[i] - testdata[j]) , (testdata[i] - testdata[j]))
    #print(i,result)
result=result/(n*n)
print(result)