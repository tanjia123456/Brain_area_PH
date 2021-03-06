from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import re
import numpy as np
import gudhi as gd

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


with open("Brain_area_triangles.txt","r",encoding="utf-8") as f:
    data = f.readlines()#根据打开的文件按行读取数据
pattern = re.compile("\[(.*?)\]")
data_triangles = []
for value in data:
    temp = pattern.findall(value.strip())[0].split(" ")
    data_triangles.append([float(i) for i in temp if i != ""])#vertices的内容，特别是坐标点的内容都是浮点数，所以用float(i)
data_triangles = np.array(data_triangles).astype(int)
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



"""这里应该有问题"""
st = gd.SimplexTree()#创建一个空的简单复合体。
#print("len(data_triangles):",len(data_triangles))#len(data_triangles): 314
for i in range(len(data_triangles)):
    #print("data_triangles[i,:]:",data_triangles[i,:])#data_triangles[i,:]: [10161  9918 10241]
    st.insert([v for v in data_triangles[i,:]], -10)##Simplicies可以挨个插入,顶点由整数索引
for i in range(len(data_vertices)):
    ##################################################################################################################
    '''这里是data_vertices[i,1]就是高度函数，因此我们这里是利用高度函数来进行过滤的'''
    st.assign_filtration([i], data_vertices[i,1])
'''＃获取所有简化程序的列表    注意，如果一条边不在顶点中，则插入边会自动插入其顶点'''
_ = st.make_filtration_non_decreasing()


'''计算过滤后的复杂体的持久性图,默认情况下，它停在1维，使用persistence_dim_max = True
计算所有维度的同源性'''
dgm = st.persistence(persistence_dim_max=True)
""" dgm = st.persistence()"""
#print("dgm:",dgm)# 有0，1，2维，0是点 1是线段 2是三角形
#但是只有一个2维  (2, (78.57119751, inf))   其余的0，1还是正常的 (1, (7.01657343, 29.42458725))  (0, (-117.12751007, inf))
#＃绘制一个持久性图
gd.plot_persistence_diagram([pt for pt in dgm if pt[0] == 0])
"""plot = gd.plot_persistence_diagram(dgm)"""
plt.title('persistence diagram with point')
plt.show()
gd.plot_persistence_diagram(dgm)
plt.title('plot_persistence_diagram')
plt.show()
gd.plot_persistence_barcode(dgm)
plt.title('plot_persistence_barcode')
plt.show()
