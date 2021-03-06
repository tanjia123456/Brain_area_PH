import numpy as np
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

label = np.loadtxt('pred_ts25.txt', dtype=int)
###列出所有的某一个脑区（脑区7）的节点索引
B = [k for k in range(len(label)) if label[k] == 7]


####提取特定脑区节点索引的Vertex
Vertices = np.loadtxt('subject26/lhvertices1.txt')
#print(Vertices, Vertices.shape)
with open("Brain_area_vertex.txt", "w") as output_file:
    for i in range(len(B)):
        output_file.write("{}\n".format(Vertices[B[i], :]))

####提取特定脑区的triangle
####应该是提取 三个都包含在特征脑区节点索引里面的  其余的没有的就是单独的点点
Triangles=np.loadtxt('subject26/lhtriangles.txt',dtype=int)
#print(Triangles,Triangles.shape)
#####去掉第一列，没用
Triangles=np.delete(Triangles, [0], axis=1)
#print(Triangles,Triangles.shape)
#对特征脑区的节点索引进行-1
B = [x-1 for x in B] # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#print("B:",B,len(B))
with open("Brain_area_triangles.txt", "w") as output_file:
    for i in range(Triangles.shape[0]):
        if set(B) > set(Triangles[i,:].tolist()):
            output_file.write("{}\n".format(Triangles[i,:]))

###构造brain_graph
with open("Brain_area_vertex.txt","r",encoding="utf-8") as f:
    data = f.readlines()#根据打开的文件按行读取数据
pattern = re.compile("\[(.*?)\]")
data_vertices = []
for value in data:
    temp = pattern.findall(value.strip())[0].split(" ")
    data_vertices.append([float(i) for i in temp if i != ""])#vertices的内容，特别是坐标点的内容都是浮点数，所以用float(i)
data_vertices = np.array(data_vertices)
print("data_vertices:",data_vertices,data_vertices.shape)


with open("Brain_area_triangles.txt","r",encoding="utf-8") as f:
    data = f.readlines()#根据打开的文件按行读取数据
pattern = re.compile("\[(.*?)\]")
data_triangles = []
for value in data:
    temp = pattern.findall(value.strip())[0].split(" ")
    data_triangles.append([float(i) for i in temp if i != ""])#vertices的内容，特别是坐标点的内容都是浮点数，所以用float(i)
data_triangles = np.array(data_triangles).astype(int)
print("data_triangles1:",data_triangles,data_triangles.shape)
#将triangle里面从小到大的数字用对应的数组进行替换
list1=[]
for i in range(data_triangles.shape[0]):
    for j in range(data_triangles.shape[1]):
        list1.append(data_triangles[i][j])
#去掉重复的数字
list2 = {}.fromkeys(list1).keys()
#进行升序排列
list3 = sorted(list2)
#A:0-n  B:升序排列的节点索引
A =  np.array(list(range(len(list3))))
B = np.array(list3)
for i in range(A.shape[0]):
    data_triangles[data_triangles==B[i]]=A[i]
print("data_triangles2:",data_triangles,data_triangles.shape)


# 创建绘图对象
'''这个转换有问题，首先vertices中是从1-264  但是triangle里面的顶点索引已经超过了'''
fig = plt.figure()
ax = fig.add_subplot(111,projection="3d")#111表明只有一个图  121就是显示在两个图中得左图  122就是显示在两个图中得右图
ax.plot_trisurf(data_vertices[:,0],data_vertices[:,1],data_vertices[:,2],triangles = data_triangles)
ax.set_zlabel('Z')  # 坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')
#ax.view_init(0, 0)#(-90, 90)
plt.title('brain_graph')
plt.savefig('brain_graph.jpg')
plt.show()







