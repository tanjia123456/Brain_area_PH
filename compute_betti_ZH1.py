import numpy as np
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#返回复数中的n-单纯形
def nSimplices(n, complex):
    nchain = []
    for simplex in complex:
        if len(simplex) == (n+1):
            nchain.append(simplex)
    if (nchain == []): nchain = [0]
    return nchain

#check if simplex is a face of another simplex
def checkFace(face, simplex):
    if simplex == 0:
        return 1
    elif set(face) < set(simplex): #if face is a subset of simplex
        return 1
    else:
        return 0
#build boundary matrix for dimension n ---> (n-1) = p
def boundaryMatrix(nchain, pchain):
    bmatrix = np.zeros((len(nchain),len(pchain)))
    i = 0
    for nSimplex in nchain:
        j = 0
        for pSimplex in pchain:
            bmatrix[i, j] = checkFace(pSimplex, nSimplex)
            j += 1
        i += 1
    return bmatrix.T


label = np.loadtxt('pred_ts25.txt', dtype=int)
#print("labels:",label,label.shape)
###列出所有的某一个脑区（脑区7）的节点索引
Vertex_id = [k for k in range(len(label)) if label[k] == 7]
#print("Vertex_id:",Vertex_id,len(Vertex_id))# 264
with open("Brain_area_Vertex_id.txt", "w") as output_file:
    for i in range(len(label)):
        #从0开始，第一个就是11   只有后面triangle和edge里面有0则不需要做任何变换
        if label[i]==7:
            output_file.write("{}\n".format(i))

def reduce_matrix(matrix):
    if np.size(matrix)==0:
        return [matrix,0,0]
    m=matrix.shape[0]
    n=matrix.shape[1]
    def _reduce(x):
        nonzero=False
        for i in range(x,m):
            for j in range(x,n):
                if matrix[i,j]==1:
                    matrix[[x,i],:]=matrix[[i,x],:]
                    matrix[:,[x,j]]=matrix[:,[j,x]]
                    nonzero=True
                    break
            if nonzero:
                break
        if nonzero:
            for i in range(x+1,m):
                if matrix[i,x]==1:
                    matrix[i,:] = np.logical_xor(matrix[x,:], matrix[i,:])
            for i in range(x+1,n):
                if matrix[x,i]==1:
                    matrix[:,i] = np.logical_xor(matrix[:,x], matrix[:,i])
            return _reduce(x+1)
        else:
            return x
    rank=_reduce(0)
    return [matrix, rank, n-rank]


Triangles=np.loadtxt('subject26/lhtriangles.txt',dtype=int)
Triangles=np.delete(Triangles, [0], axis=1)
with open("Brain_area_triangles.txt", "w") as output_file:
    for i in range(Triangles.shape[0]):
        if set(Vertex_id) > set(Triangles[i,:].tolist()):
            output_file.write("{}\n".format(Triangles[i,:]))
#去掉其左右的"[ ]"
with open("Brain_area_triangles.txt","r",encoding="utf-8") as f:
    data = f.readlines()#根据打开的文件按行读取数据
pattern = re.compile("\[(.*?)\]")
data_triangles = []
for value in data:
    temp = pattern.findall(value.strip())[0].split(" ")
    data_triangles.append([float(i) for i in temp if i != ""])#vertices的内容，特别是坐标点的内容都是浮点数，所以用float(i)
data_triangles = np.array(data_triangles).astype(int)
#print("data_sanjiao:",data_triangles)


#若是每行（边）的节点索引都属于  特定脑区的节点索引里面，那就保留下那一行
Edges=np.loadtxt('subject26/lhedges.txt',dtype=int)
with open("Brain_area_edges.txt", "w") as output_file:
    for i in range(Edges.shape[0]):
        if set(Vertex_id) > set(Edges[i,:].tolist()):
            output_file.write("{}\n".format(Edges[i,:]))
with open("Brain_area_edges.txt","r",encoding="utf-8") as f:
    data = f.readlines()#根据打开的文件按行读取数据
pattern = re.compile("\[(.*?)\]")
data_edges = []
for value in data:
    temp = pattern.findall(value.strip())[0].split(" ")
    data_edges.append([float(i) for i in temp if i != ""])#vertices的内容，特别是坐标点的内容都是浮点数，所以用float(i)
data_edges = np.array(data_edges).astype(int)
#print("data_edge:",data_edges)



###首先是得到0维,1维,2维单形      ###0维就是节点索引，一维就是边的关系，二维就是triangle
complex_0=np.loadtxt('Brain_area_Vertex_id.txt').astype(int)
b=[]
for i in range(complex_0.shape[0]):
    b.append({complex_0[i]})#[{10237}, {10239}, {10240}, {10241}]
complex_0=b
#print(b,type(b))
complex_1=data_edges.tolist()
complex_2=data_triangles.tolist()
#print("complex_0:",complex_0)
complexs=complex_0+complex_1+complex_2
#print("complexs:",complexs,type(complexs))
#[1,2,3,[1,2],[1,2,3]]--->[{1},{2},{3},{1,2},{1,2,3}]



###计算C0  264
chain0 = nSimplices(0, complexs)#
####计算C1  1412
chain1 = nSimplices(1, complexs)#
###计算c2 441
chain2 = nSimplices(2,complexs)

#节点264  :边1412  三角441
###相当于chanin1:264   chanin2:1412  chain3:441
Bn1=boundaryMatrix(chain1, chain0)# (264, 1412)
Bn2=boundaryMatrix(chain1,chain2)#边数、三角数--->[三角数，边数，进行一个转置了]   (441, 1412)


#Initialize boundary matrices
####点的个数
boundaryMap0 = np.matrix(np.zeros(264).astype(int))
boundaryMap1 = np.matrix(Bn1)#节点，边数
boundaryMap2 = np.matrix(Bn2)#三角数，边数


#Smith normal forms of the boundary matrices
smithBM0 = reduce_matrix(boundaryMap0)
smithBM1 = reduce_matrix(boundaryMap1)
smithBM2 = reduce_matrix(boundaryMap2)

#Calculate Betti numbers
betti0 = (smithBM0[2] - smithBM1[1])#0-1  位置都是2,1
betti1 = (smithBM1[2] - smithBM2[1])#2-1 位置都是2,1
betti2 = 0  #There is no n+1 chain group, so the Betti is 0


print(" Betti #0: %s \n Betti #1: %s \n Betti #2: %s" % (betti0, betti1, betti2))
'''
Betti #0: 2 
Betti #1: 1150 
Betti #2: 0
感觉这个结果就不合理，所以应该是有问题的...具体问题出在哪里不太清楚
'''























