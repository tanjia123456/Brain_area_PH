from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import re
import numpy as np
import gudhi as gd

label = np.loadtxt('pred_ts25.txt', dtype=int)
B = [k for k in range(len(label)) if label[k] == 7]

####提取特定脑区节点索引的Vertex
Vertices = np.loadtxt('lhvertices1.txt')
with open("Brain_area_vertex.txt", "w") as output_file:
    for i in range(len(B)):
        output_file.write("{}\n".format(Vertices[B[i], :]))

Triangles=np.loadtxt('lhtriangles.txt',dtype=int)#
Triangles=np.delete(Triangles, [0], axis=1)
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
data_vertices = np.array(data_vertices)#(264,3)


with open("Brain_area_triangles.txt","r",encoding="utf-8") as f:
    data = f.readlines()#根据打开的文件按行读取数据
pattern = re.compile("\[(.*?)\]")
data_triangles = []
for value in data:
    temp = pattern.findall(value.strip())[0].split(" ")
    data_triangles.append([float(i) for i in temp if i != ""])#vertices的内容，特别是坐标点的内容都是浮点数，所以用float(i)
data_triangles = np.array(data_triangles).astype(int)#print("data_triangle1:",data_triangles,data_triangles.shape)#data_triangle: (441, 3)

#将triangle里面从小到大的数字用对应的数组进行替换
list1=[]
for i in range(data_triangles.shape[0]):
    for j in range(data_triangles.shape[1]):
        list1.append(data_triangles[i][j])
list2 = {}.fromkeys(list1).keys()
list3 = sorted(list2)
C =  np.array(list(range(len(list3))))
D = np.array(list3)
for i in range(C.shape[0]):
    data_triangles[data_triangles==D[i]]=C[i]#print("data_triangle2:",data_triangles,data_triangles.shape)#data_triangle: (441, 3)

# create barin area graph
fig = plt.figure()
ax = fig.add_subplot(111,projection="3d")#111表明只有一个图  121就是显示在两个图中得左图  122就是显示在两个图中得右图
ax.plot_trisurf(data_vertices[:,0],data_vertices[:,1],data_vertices[:,2],triangles = data_triangles)
ax.set_zlabel('Z')  # 坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')
plt.title('brain_graph')
plt.savefig('brain_graph.jpg')
plt.show()

####the value of curv as filtration
sulc=np.loadtxt('lhcurvfeatures.txt',dtype=float)
with open("Brain_area_sulc.txt", "w") as output_file:
    for i in range(len(B)):
        output_file.write("{}\n".format(sulc[B[i],1]))
filteration_value=np.loadtxt('Brain_area_sulc.txt',dtype=float)##print(filteration_value.shape)#(264,)


st = gd.SimplexTree()#创建一个空的简单复合体。
for i in range(len(data_triangles)):
    st.insert([v for v in data_triangles[i,:]], -10)
    ####can't feed the filtration value(the curv of every node)
    ####error message:Process finished with exit code -1073741819 (0xC0000005)
for i in range(len(data_vertices)):
    st.assign_filtration([i], filtration = filteration_value[i])


st_gen = st.get_filtration()
for splx in st_gen:
    print("splx:",splx)


dgm = st.persistence(persistence_dim_max=True)
gd.plot_persistence_diagram([pt for pt in dgm if pt[0] == 0])
plt.show()
gd.plot_persistence_diagram(dgm)
plt.show()
gd.plot_persistence_barcode(dgm)
plt.show()






