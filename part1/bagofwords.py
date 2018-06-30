from scipy import sparse
import numpy as np
import csv
def getFeature(comment): # read a comment
    row=[]
    col=[]
    data=[]
    comment=comment[1:]
    for i in range(len(comment)):
        index_value=comment[i].split(":")
        index=int(index_value[0])
        value=int(index_value[1])
        col.append(index)
        data.append(value)
        row.append(0)
    vcom=sparse.coo_matrix((data,(row,col)),shape=(1,89527))
    v=np.array(vcom.toarray())
    #v = preprocessing.normalize(v, norm='l1')
    return v[0]

word=np.zeros(89527)
toatal=np.ones(89527)
toatal=toatal*50000
with open("data","r") as file:
    for line in file.readlines():
        line = line.strip().split(' ')
        v = getFeature(line)
        word = word+np.where(v!=0, 1, 0)

idf = np.log(np.true_divide(toatal,word))
out = open("idf.txt", 'w')
csv_writer = csv.writer(out)
csv_writer.writerow(idf)

