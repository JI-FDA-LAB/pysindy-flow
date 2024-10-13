import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import random
import os
import math
# 计算两个点之间的欧式距离，参数为两个元组
def dist(t1, t2):
    dis = math.sqrt((np.power((t1[0]-t2[0]),2) + np.power((t1[1]-t2[1]),2)))
    # print("两点之间的距离为："+str(dis))
    return dis
 

# DBSCAN算法，参数为数据集，Eps为指定半径参数，MinPts为制定邻域密度阈值
def dbscan(Data, Eps, MinPts):
    num = len(Data)  # 点的个数
    # print("点的个数："+str(num))
    unvisited = [i for i in range(num)]  # 没有访问到的点的列表
    # print(unvisited)
    visited = []  # 已经访问的点的列表
    C = [-1 for i in range(num)]
    # C为输出结果，默认是一个长度为num的值全为-1的列表
    # 用k来标记不同的簇，k = -1表示噪声点
    k = -1
    # 如果还有没访问的点
    while len(unvisited) > 0:
        # 随机选择一个unvisited对象
        p = random.choice(unvisited)
        unvisited.remove(p)
        visited.append(p)
        # N为p的epsilon邻域中的对象的集合
        N = []
        for i in range(num):
            if (dist(Data[i], Data[p]) <= Eps):# and (i!=p):
                N.append(i)
        # 如果p的epsilon邻域中的对象数大于指定阈值，说明p是一个核心对象
        if len(N) >= MinPts:
            k = k+1
            # print(k)
            C[p] = k
            # 对于p的epsilon邻域中的每个对象pi
            for pi in N:
                if pi in unvisited:
                    unvisited.remove(pi)
                    visited.append(pi)
                    # 找到pi的邻域中的核心对象，将这些对象放入N中
                    # M是位于pi的邻域中的点的列表
                    M = []
                    for j in range(num):
                        if (dist(Data[j], Data[pi])<=Eps): #and (j!=pi):
                            M.append(j)
                    if len(M)>=MinPts:
                        for t in M:
                            if t not in N:
                                N.append(t)
                # 若pi不属于任何簇，C[pi] == -1说明C中第pi个值没有改动
                if C[pi] == -1:
                    C[pi] = k
        # 如果p的epsilon邻域中的对象数小于指定阈值，说明p是一个噪声点
        else:
            C[p] = -1
 
    return C


def binary_matrix_index(matrix,vx,vy):
    rows, cols = np.where(matrix == 1)
    return np.column_stack((rows, cols))


def batch_vortex_identification (vx,vy,n):
    '''
    x_file_path = './flow_field/'+str(wc)+'/x/'+str(cycle)+'-'+str(time)+'.csv';
    y_file_path = './flow_field/'+str(wc)+'/y/'+str(cycle)+'-'+str(time)+'.csv';
    # y_file_path = './y/'+str(wc)+'-'+str(cycle)+'-'+str(time)+'.csv';

    # Read the csv files using pandas.
    # vx = pd.read_csv('x.csv', header=None)
    # vy = pd.read_csv('y.csv', header=None)
    # vx = pd.read_csv(x_file_path, header=None)
    # vy = pd.read_csv(y_file_path, header=None)

    vx = pd.read_csv(x_file_path, header=None)
    vy = pd.read_csv(y_file_path, header=None)
    vx= vx.iloc[2:47,12:22]
    vy= vy.iloc[2:47,12:22]
    '''
    # Create a meshgrid using numpy.
    x, y = np.meshgrid(np.arange(0, n), np.arange(0, n))

    # print(vx.values)

    # print(np.gradient(vy.values))
    # Calculate the vorticity using numpy.

    # Identify the vortex core using the delta criterion.
    delta = vx**2 + vy**2
    # P = np.gradient(vy.values)[1]
    # Q = np.gradient(vx.values)[2]
    P = np.gradient(vx)[0] + np.gradient(vy)[1]
    Q = np.gradient(vx)[1]*np.gradient(vy)[0] - np.gradient(vx)[0]*np.gradient(vy)[1]

    delta = P**2 - 4*Q
    df_delta=pd.DataFrame(delta)


    df_delta.to_csv('delta.csv', index=False)
    
    x_index, y_index = np.where(df_delta<0)
    '''
    plt.figure(figsize=(8, 6), dpi=80)
    plt.scatter(x_index,y_index, marker='^')
    plt.show()
    '''
    vortex_core = delta < -0.5

    # print(vortex_core)

    vorticity = (np.gradient(vy.values)[1] - np.gradient(vx.values)[0])


    # dis = dist((1,1),(3,4))
    # print(dis)


    core_ind = binary_matrix_index(vortex_core, vx, vy)
    C = dbscan(core_ind, 4, 13)
    # print(C)
    xcls = []
    ycls = []
    for data in core_ind:
        xcls.append(data[0])
        ycls.append(data[1])
    # plt.figure(figsize=(8, 6), dpi=80)
    # plt.scatter(ycls,xcls, c=C, marker='^')
    # plt.show()

    clusters = []
    for i in range(max(C)+1):
        cluster = []
        for j in range(len(C)):
            if C[j] == i:
                cluster.append([xcls[j], ycls[j]])
        if len(cluster) > 0:
            clusters.append(np.array(cluster))
            


    # Calculate the circulation and circle for each cluster of vortex core.
    # plt.figure(figsize=(5, 15))
    #result = pd.DataFrame({'center_x': [], 'center_y': [],'adj_center_x': [], 'adj_center_y': [], 'radius':[], 'strength':[], 'shape':[]})


    for cluster in clusters:
        if len(cluster) > 10:
            # cluster_vorticity = vorticity[min(cluster[0]):max(cluster[0])+1, min(cluster[1]):max(cluster[1])+1]
            cluster_vorticity = vorticity[(cluster)]
            circulation = np.sum(cluster_vorticity)
            center_y, center_x = np.mean(cluster, axis=0)
            radius = np.sqrt(np.sum((cluster - [center_y, center_x])**2)) / np.sqrt(len(cluster))
            # print(abs(1 - (len(cluster) / (np.pi * radius ** 2))))
            # print(radius)
            similarity_measurement = 'circle' if abs(1 - (len(cluster) / (np.pi * radius ** 2))) < 0.4 else 'not circle'
            

            #draw_circle = plt.Circle((center_x, center_y), radius, color='b', fill=False, linestyle='--')
            # plt.text(center_x+radius*1.2, center_y-5, label,color='purple')
            #plt.scatter(center_x,center_y,c='none', marker = 'v', edgecolors = 'blue',s=5)
            mat = np.ones((n,n))
            for i in range(1,n-1):
                for j in range(1,n-1):
                    mat[i][j] = calculate_average_SI_index(vx.values, vy.values, i, j)
            adj_center_x, adj_center_y = find_closest_point(mat, center_y, center_x, radius)
            #plt.scatter(adj_center_x,adj_center_y,c='none', marker = 'v', edgecolors = colorr,s=5)
            
            #draw_adj_circle = plt.Circle((adj_center_x, adj_center_y), radius, color=colorr, fill=False, linestyle='--')
            #ax.add_artist(draw_circle)
            #ax.add_artist(draw_adj_circle)
            #label = f'Ensembled Center: ({center_x:.2f}, {center_y:.2f}), \n Adjusted Center: ({adj_center_x:.2f}, {adj_center_y:.2f}), \n Radius: {radius:.2f}, \n Strength: {circulation:.2f}, \n Shape: {similarity_measurement}'
            #if colorr=='r':
                #plt.text(25,25, label,color=colorr)
            #else:
                #plt.text(25,0, label,color=colorr)
            # Add a new row to the DataFrame
            #new_row = {'center_x': center_x, 'center_y': center_y,'adj_center_x': adj_center_x, 'adj_center_y': adj_center_y, 'radius':radius, 'strength':circulation, 'shape':similarity_measurement}
            #result = result.append(new_row, ignore_index=True)
    # Visualize the flow field with the vortex core labeled in triangles.

    #plt.quiver(x,y,vx, vy,color=colorr)
    # for cluster in clusters:
    #     if len(cluster) > 10:
    #         plt.triplot(cluster[:, 1], cluster[:, 0], color='r')
    #plt.scatter(ycls,xcls, c=C, marker='^',s=20,alpha=0.35)

    #plt.axis([-1,10,0,50])
    #plt.xlim(-2,10)
    '''
    df_x=pd.DataFrame(x)
    df_y=pd.DataFrame(y)
    df_x.to_csv('upx.csv', index=False)
    df_y.to_csv('upy.csv', index=False)
    np.savetxt('upxx.txt',x)
    '''
    return [adj_center_x, adj_center_y, radius], circulation

'''
    fig_save_path =  './vortex_identification/'+str(wc)+'/'
    fig_save_name = fig_save_path +str(cycle)+'-'+str(time)+'.png';
    
    res_save_name = fig_save_path +str(cycle)+'-'+str(time)+'.csv';
    
    if not os.path.exists(fig_save_path):
        os.mkdir(fig_save_path)
    # print(calculate_SI_index(vx.values,vy.values))

    result.to_csv(res_save_name)
    plt.savefig(fig_save_name,dpi=300)
    # print(cluster)
'''
    # plt.figure(figsize=(5, 15))
    # plt.contourf(mat)
    
    
    
    
#     vorticity = np.gradient(vy, axis=0) - np.gradient(vx, axis=1)
#     find_closest_point(vorticity, center_y, center_x, radius)

#     # Plot the vorticity map
#     plt.figure(figsize=(5, 15))
#     plt.contourf( vorticity)
#     plt.colorbar()
#     plt.show()


def calculate_SI_index(v, vi):
    return np.dot(v, vi) / (np.linalg.norm(v) * np.linalg.norm(vi))

def calculate_average_SI_index(vx, vy, i, j):

    si_indices = []
    v = np.array([vx[i][j], vy[i][j]])
    vi = np.array([[vx[i-1][j-1], vy[i-1][j-1]],
                   [vx[i-1][j], vy[i-1][j]],
                   [vx[i-1][j+1], vy[i-1][j+1]],
                   [vx[i][j-1], vy[i][j-1]],
                   [vx[i][j+1], vy[i][j+1]],
                   [vx[i+1][j-1], vy[i+1][j-1]],
                   [vx[i+1][j], vy[i+1][j]],
                   [vx[i+1][j+1], vy[i+1][j+1]]])
    for k in range(8):
        if np.linalg.norm(vi[k]) == 0 or np.linalg.norm(v)==0:
            continue;
        res = calculate_SI_index(v, vi[k])-np.linalg.norm(vi[k])/10;
        
        si_indices.append(res)
        
    if si_indices == []:
        si_indices.append(1)
    return np.average(si_indices)


def find_closest_point(vorticity_map, center_x, center_y, radius):
    # Create a mask for the circle
    y,x = np.ogrid[-center_x:vorticity_map.shape[0]-center_x, -center_y:vorticity_map.shape[1]-center_y]
    mask = x*x + y*y <= radius*radius

    masked_vorticity_map = np.ma.masked_array(vorticity_map, ~mask)
    
    closest_point = np.unravel_index(masked_vorticity_map.argmin(), masked_vorticity_map.shape)

    return closest_point[1],closest_point[0];

def cmp_center(v_x,v_y,u_x,u_y,n):
    center=np.zeros(6)
    strength=np.zeros(2)
    #fig, ax = plt.subplots(figsize=(5, 5))
    
    
    df_vx=pd.DataFrame(v_x)
    df_vy=pd.DataFrame(v_y)
    center[:3],strength[0] = batch_vortex_identification(df_vx,df_vy,n)
    
    df_vx=pd.DataFrame(u_x)
    df_vy=pd.DataFrame(u_y)
    center[3:],strength[1] = batch_vortex_identification(df_vx,df_vy,n)
    bias_center=dist([center[0],center[1]],[center[3],center[4]])
    #plt.show
    return center, bias_center, strength
    