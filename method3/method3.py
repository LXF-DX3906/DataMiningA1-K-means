#!/usr/bin/env python
# coding: utf-8

# In[1]:


# step1: 数据预处理
# customer_pur_recoder={vipno:[purTime，pluno, category:[]]}

# step2: 对于每个客户，构造FTCTree
# ftctree_dict={vipno:FTCTree}

# //实现GetCT函数
# //由几个更小的函数组成
# step3: 求unionTree（利用迭代，temp_unionTree与新客户FTCTree进行union即可）

# step4: 求AvgFreq(utree)和MaxFreq(utree)

# step5: 实现update(utree,freq)函数

# step6: 实现Dist(FTCTree,utree)函数

# //实现BIC
# step7: 实现簇的分裂算法（Kmeans）

# step8: 实现求bic函数，对于分裂前后的簇，分别求bic


# ## 数据预处理，得到数据：customer_pur_recoder = { vipno: \[purTime，pluno, category:\[ \] \]}

# In[89]:


import pandas as pd
import json
import datetime
from IPython.display import HTML
from treelib import Node, Tree
from numpy import mean
import random
from copy import deepcopy
import math
import operator


# In[3]:


df = pd.read_csv('../trade_new.csv')

# 取出所需要的信息并根据vipno排序
useful_info = df[['vipno', 'pluno', 'sldatime']].sort_values('vipno')
useful_info.reset_index(drop=True, inplace=True)
HTML(useful_info.to_html(index=False,max_rows=100))


# In[4]:


# 取出所有的vipno列表（去重）
vipno = useful_info.drop_duplicates(subset=['vipno'], keep='first', inplace=False)['vipno']
# vipno


# In[5]:


customer_pur_recoder = {}
# 以每个vipno为key，值为{pluno: [], sldatime: []，category: []}
for item in vipno:
    customer_pur_recoder[item]={}
    customer_pur_recoder[item]['pluno'] = useful_info.loc[useful_info['vipno'] == item]['pluno'].tolist()
    customer_pur_recoder[item]['sldatime'] = useful_info.loc[useful_info['vipno'] == item]['sldatime'].tolist()
    customer_pur_recoder[item]['category'] = []
# customer_pur_recoder


# In[6]:


# 求所有客户中最近一次的购买时间
t_df = df['sldatime'].sort_values(ascending = False)
t_df.reset_index(drop=True, inplace=True)
latest_time = datetime.datetime.strptime(t_df[0],'%Y-%m-%d %H:%M:%S')
# latest_time


# In[7]:


# 得到category
# 遍历所有客户
for index in customer_pur_recoder:
    # 对每个客户，遍历所有购买记录
    customer_pur_recoder[index]['category'] = []
    for index2,item in enumerate(customer_pur_recoder[index]['sldatime']):
        t_category = []
        temp_time = datetime.datetime.strptime(item,'%Y-%m-%d %H:%M:%S')
        diff_days = (latest_time - temp_time).days
        t_category.append(int(customer_pur_recoder[index]['pluno'][index2]/1000000))
        if diff_days <= 120:
            t_category.append(int(customer_pur_recoder[index]['pluno'][index2]/100000))        
        if diff_days <= 60:
            t_category.append(int(customer_pur_recoder[index]['pluno'][index2]/10000))
        if diff_days <= 30:
            t_category.append(int(customer_pur_recoder[index]['pluno'][index2]/1000))
        customer_pur_recoder[index]['category'].append(t_category.copy())
# customer_pur_recoder


# ## 对于每个客户，构造FTCTree

# In[8]:


# 定义节点 node = {clifi:pluno/10**(7-d),freq:n} d为节点的深度
ftctree_dict = {}
# i = 0
for index in customer_pur_recoder:
#     if i > 0:
#         break
#     i+=1
    c_tree = Tree()
    c_tree.create_node(index,'root')
    for category in customer_pur_recoder[index]['category']:
        parent = 'root'
        for item in category:
            if c_tree.contains(item):
                c_tree[item].data += 1
            else:
                c_tree.create_node(item, item, data = 1, parent= parent)
            parent = item
    ftctree_dict[index] = c_tree
# c_tree.show()
# ftctree_dict
# print(c_tree.to_json(with_data=True))


# ## 求unionTree

# In[46]:


# 781924   13325038116
# c_tree1 = ftctree_dict[781924]
# c_tree2 = ftctree_dict[13325038116]
# c_tree1.show()
# c_tree2.show()


# In[50]:


# 求unionTree（利用迭代，temp_unionTree与新客户FTCTree进行union即可）
def union_tree(c_tree1, c_tree2):
    union_tree = Tree(tree=c_tree1, deep=True)
    for item in c_tree2.expand_tree('root'):
        if item == "root":
            continue
        if union_tree.contains(item):
            union_tree[item].data += c_tree2[item].data
        else:
            parent = c_tree2.parent(item).identifier
            union_tree.create_node(item, item, data = c_tree2[item].data, parent = parent)
    return union_tree

def inter_tree(c_tree1, c_tree2):
    u_tree = union_tree(c_tree1, c_tree2)
    inter_tree = Tree()
    inter_tree.create_node('ROOT','root')
    for item in u_tree.expand_tree('root'):
        if item == 'root':
            continue
        if c_tree1.contains(item) and c_tree2.contains(item):
            parent = u_tree.parent(item).identifier
            inter_tree.create_node(item, item, data = u_tree[item].data, parent = parent)
    return inter_tree

def cluster_union_tree(l_ftctree):
    if len(l_ftctree) == 0:
        return None
    elif len(l_ftctree) == 1:
        return l_ftctree[0]
    elif len(ftctree_dict) == 2:
        return union_tree(l_ftctree[0], l_ftctree[1])
    else:
#       cluster_union_tree = union_tree(ftctree_dict[list(ftctree_dict.keys())[0]], ftctree_dict[list(ftctree_dict.keys())[1]])
        temp_list = []
        is_odd = len(l_ftctree)%2
        if is_odd != 0: 
            temp_list.append(l_ftctree[0])
        for index, item in enumerate(l_ftctree):
            if is_odd != 0:
                if index % 2 == 0 and index != 0:
                    temp_list.append(union_tree(l_ftctree[index-1],l_ftctree[index]))
            else:
                if index % 2 == 1:
                    temp_list.append(union_tree(l_ftctree[index-1],l_ftctree[index]))
        c_union_tree = cluster_union_tree(temp_list)
        return c_union_tree


# In[52]:


# i_tree = inter_tree(c_tree1, c_tree2)
# i_tree.show()
# u_tree =  union_tree(c_tree1, c_tree2)
# u_tree.show()
# pa = u_tree.parent(item).identifier


# In[28]:


# 测试cluster_union_tree函数
# t_ftc = {}
# for index, item in enumerate(list(ftctree_dict.keys())):
#             if index < 400:
#                 t_ftc[item] = ftctree_dict[item]
#             else:
#                 break
#
# test_u_tree = cluster_union_tree(list(t_ftc.values()))
# print(test_u_tree.to_json(with_data=True))


# In[29]:


# test_u_tree.show()


# In[30]:


# test_u_tree.children('root')


# In[31]:


# 求AvgFreq(utree)和MaxFreq(utree)
def AvgFreq(utree):
    freq_list = []
    for item in utree.expand_tree('root'):
        if item == "root":
            continue
        freq_list.append(utree[item].data)
    return mean(freq_list)

# avg = AvgFreq(test_u_tree)
# avg


# In[32]:



def MaxFreq(utree):
    freq_list = []
    for item in utree.expand_tree('root'):
        if item == "root":
            continue        
        freq_list.append(utree[item].data)
    m = max(freq_list)
    return m

# m = MaxFreq(test_u_tree)
# m


# In[33]:


# 实现update(utree,freq)函数
def update(utree, freq):
    updt_tree = Tree(tree=utree,deep=True)
    for item in utree.expand_tree('root'):
        if item == 'root':
            continue
        if updt_tree.contains(item) and utree[item].data < freq:
            updt_tree.remove_node(item)
    return updt_tree

# updt_tree = update(test_u_tree, 3)
# updt_tree.show()


# In[34]:


# itree = Tree()
# itree.create_node(tag="INTER", identifier="root",data= 8)
# itree.create_node(15, 15, data = 5, parent="root")
# itree.create_node(12, 12, data = 3, parent="root")
# itree.create_node(151, 151, data = 3, parent=15)
# itree.create_node(152, 152, data = 2, parent=15)
# itree.create_node(121, 121, data = 3, parent=12)
# itree.create_node(1211, 1211, data = 3, parent=121)
# itree.show()


# ## 实现Dist(FTCTree,utree)函数

# In[35]:


# 求树某深度的所有节点
def level_node(tree,n):
    lev_n_list = []
    for item in tree.expand_tree('root'):
        if tree.level(item) == n:
            lev_n_list.append(item)
    return lev_n_list


# 求每一层的每个节点的Sim（v,utree）
# sim_node_dic = {lev1:{identifi1: sim}}
def sim_node(ftctree,utree):
    sim_node_dic = {}
    for dp in range(1,ftctree.depth()+1):
        sim_node_dic[dp] = {}
        ftc_dp_list = level_node(ftctree, dp)
        if dp == 1:
            union_dp_list = level_node(utree, dp)
            union_dp_freq_sum = 0
            for node_inden in union_dp_list:
                union_dp_freq_sum += utree[node_inden].data
            for node_inden in ftc_dp_list:
                sim_node_dic[dp][node_inden] = ftctree[node_inden].data/union_dp_freq_sum
        else:
            for node_inden in ftc_dp_list:
                sim_node_dic[dp][node_inden] = ftctree[node_inden].data/utree.parent(node_inden).data
    return sim_node_dic


    # 求每一层的Sim（Vl,tree）
def sim_lev(ftctree, sim_node_dic):
    sim_lev_dic = {}
    for lev in sim_node_dic:
        if lev == 1:
            sim_lev_dic[lev] = sum(list(sim_node_dic[lev].values()))
        else:
            l_par = []
            for index in sim_node_dic[lev]:
                l_par.append(ftctree.parent(index))
            l_par = list(set(l_par))
            par_num = len(l_par)
            # sim_lev_dic[lev] = sum(list(sim_node_dic[lev].values()))/len(sim_node_dic[lev])
            sim_lev_dic[lev] = sum(list(sim_node_dic[lev].values())) / par_num
    return sim_lev_dic


def Dist(ftctree,utree):
    sim_node_dic = sim_node(ftctree,utree)
    sim_lev_dic = sim_lev(ftctree,sim_node_dic)
    dist = 1
    lev_sum = sum(list(sim_lev_dic.keys()))
    for lev in sim_lev_dic:
        dist -= sim_lev_dic[lev]*lev/lev_sum
    return dist


# dist = Dist(itree,itree)
#
# dist


# In[36]:


# utree = cluster_union_tree(list(ftctree_dict.values()))
# utree
# dist = Dist(inter_tree(cluster[index],utree),union_tree(cluster[index],utree))


# ## 实现GetCT（cluster = ftctree_dict）

# In[53]:


# 实现GetCT（cluster = ftctree_dict）
def GetCT(cluster):
    utree = cluster_union_tree(list(cluster.values()))
    freq = 1
    mindist = float('Inf')
    freqStep = AvgFreq(utree)
    freqEnd = MaxFreq(utree)
    # num_sum_nodes = 0
    # for item in list(cluster.values()):
    #     num_sum_nodes += len(item.all_nodes())
    # num_avg_nodes = num_sum_nodes/len(list(cluster.values()))
    while freq <= freqEnd:
        utree = update(utree, freq)
        dist = 0
        for index in cluster:
            dist += Dist(inter_tree(cluster[index],utree),union_tree(cluster[index],utree))
        if dist < mindist:
            mindist = dist
            ct = utree
            # if len(ct.all_nodes()) <= num_avg_nodes:
            #     break
        freq = freq + freqStep
    return ct
# ct = GetCT(ftctree_dict)
# ct


# In[54]:


# ct.show()


# ## 实现簇的分裂算法

# In[71]:


# KMeans K=2

# 1 初始化质心（随机选取两个用户点随机选择N对值，选取一对距离最大的质心。）
#   缺点：选择的用户树可能数据量很小。不足以代表，可添加约束条件，选择的种子树结点个数不小于平均结点个数。
#   初始化簇，开始时簇中包含两质心
# 2 计算每个用户点到质心的距离，比较后加入距离较短的簇中
# 3 更新质心，判断两个簇是否发生变化，变化则执行2，不变则结束算法

# 计算簇中所有树的平均节点个数
def avg_nodes_num(ftctree_dict):
    sum_nodes_num = 0
    for index in ftctree_dict:
        sum_nodes_num += len(ftctree_dict[index].all_nodes())
    return sum_nodes_num/len(ftctree_dict)

# 初始化质心
def calculate_initial_centroids_first_run(ftctree_dict):
    avg_n_sum = avg_nodes_num(ftctree_dict)
    init_centroids_list = []
    init_dist_list = []
    i = 0
    while i < 10:
        temp_c = random.sample(list(ftctree_dict.keys()), 2)
        if len(ftctree_dict[temp_c[0]]) >= avg_n_sum and len(ftctree_dict[temp_c[1]]) >= avg_n_sum:
            init_centroids_list.append(temp_c)
            i += 1
    for item in init_centroids_list:
        init_dist_list.append(Dist(inter_tree(ftctree_dict[item[0]], ftctree_dict[item[1]]), union_tree(ftctree_dict[item[0]], ftctree_dict[item[1]])))
    max_dist_index = init_dist_list.index(max(init_dist_list))
    return {init_centroids_list[max_dist_index][0]:ftctree_dict[init_centroids_list[max_dist_index][0]], init_centroids_list[max_dist_index][1]:ftctree_dict[init_centroids_list[max_dist_index][1]]}

# centroids = calculate_initial_centroids_first_run(ftctree_dict)
# centroids
# a = init_centroids[list(init_centroids.keys())[0]]
# b = init_centroids[list(init_centroids.keys())[1]]
# Dist(inter_tree(a,b), union_tree(a,b))


# In[70]:


# # 初始化簇，开始时簇中包含两质心
# def init_cluster(centroids):
#     cluster = {}
#     ct1 = list(centroids.keys())[0]
#     ct2 = list(centroids.keys())[1]
#     cluster[ct1] = [ct1]
#     cluster[ct2] = [ct2]
#     return cluster


# In[74]:


# 计算每个用户点到质心的距离，比较后加入距离较短的簇中(更新簇)
def calculate_cluster(ftctree_dict, centroids):
    cluster = {}
    ct1 = list(centroids.keys())[0]
    ct2 = list(centroids.keys())[1]
    cluster[ct1] = []
    cluster[ct2] = []
    ct1_tree = centroids[ct1]
    ct2_tree = centroids[ct2]
    for index in ftctree_dict:
        dist1 = Dist(inter_tree(ftctree_dict[index],ct1_tree), union_tree(ftctree_dict[index],ct1_tree))
        dist2 = Dist(inter_tree(ftctree_dict[index],ct2_tree), union_tree(ftctree_dict[index],ct2_tree))
        if dist1 > dist2:
            cluster[ct2].append(index)
        else:
            cluster[ct1].append(index)
    return cluster

# cluster = calculate_cluster(ftctree_dict, centroids)
# cluster


# In[77]:


# 更新质心
def update_centroids(ftctree_dict, cluster):
    ct1 = list(cluster.keys())[0]
    ct2 = list(cluster.keys())[1]
    cluster1_dict = {}
    cluster2_dict = {}
    for item in cluster[ct1]:
        cluster1_dict[item] = ftctree_dict[item]
    for item in cluster[ct2]:
        cluster2_dict[item] = ftctree_dict[item]
    centroids = {}
    centroids['ct1'] = GetCT(cluster1_dict)
    centroids['ct2'] = GetCT(cluster2_dict)
    return centroids
# centroids = update_centroids(ftctree_dict, cluster)
# centroids


# In[94]:


# centroids['ct1'].show()
# centroids['ct2'].show()


# In[79]:


# 判断簇是否发生变化
def is_change(last_cluster, cluster):
    equ_n = 0
    for index in cluster:
        for index2 in last_cluster:
            if operator.eq(cluster[index], last_cluster[index2]):
                equ_n += 1
    if equ_n == 2:
        return False
    return True


# In[91]:


# 实现Kmeans
def Kmeans(cluster_dict):
    print('初始化质心')
    centroids = calculate_initial_centroids_first_run(cluster_dict)
    print('根据初始化质心计算簇')
    cluster = calculate_cluster(cluster_dict, centroids)
    last_cluster = deepcopy(cluster)
    isChange = True
    i = 0
    while isChange:
        i+=1
        print('第'+str(i)+'次更新质心')
        centroids = update_centroids(cluster_dict, cluster)
        centroids['ct1'].show()
        centroids['ct2'].show()
        print('第'+str(i)+'次计算簇')
        cluster = calculate_cluster(cluster_dict, centroids)
        print('cluster1: ' + str(len(cluster['ct1'])))
        print(cluster['ct1'])
        print('cluster2: ' + str(len(cluster['ct2'])))
        print(cluster['ct2'])
        isChange = is_change(last_cluster, cluster)
        last_cluster = deepcopy(cluster)
    result = {}
    result['cluster'] = cluster
    result['centroids'] = centroids
    return result


# In[92]:


# 实现求bic函数，对于分裂前后的簇，分别求bic
def calculate_sigma(cluster_dict, ct_tree, clu_num):
    dist = 0
    for index in cluster_dict:
        dist += Dist(inter_tree(cluster_dict[index],ct_tree),union_tree(cluster_dict[index],ct_tree))**2
    sigma = (1/(len(cluster_dict) - clu_num))*dist
    return sigma

def plu_num(cluster_dict):
    l_p = []
    for index in cluster_dict:
        l_leaves = cluster_dict[index].leaves()
        for item in l_leaves:
            if cluster_dict[index].depth(item) == 4:
                l_p.append(item.identifier)
    return len(list(set(l_p)))
            

def calculate_likelihood_before(cluster_dict, ct_tree, clu_num):
    c_num = len(cluster_dict)
    likhood = c_num*math.log(c_num) - c_num*math.log(c_num) - (c_num/2)*math.log(2*math.pi) \
              - (c_num*plu_num(cluster_dict)/2)*math.log(calculate_sigma(cluster_dict, ct_tree, clu_num)) - (c_num - clu_num)/2
    return likhood

def calculate_likelihood_after(cluster_dict,cluster_dict1,ct1_tree,cluster_dict2,ct2_tree,clu_num):
    c_num1 = len(cluster_dict1)
    likhood = c_num1*math.log(c_num1) - c_num1*math.log(len(cluster_dict)) - (c_num1/2)*math.log(2*math.pi) \
              - (c_num1*plu_num(cluster_dict1)/2)*math.log(calculate_sigma(cluster_dict1, ct1_tree, clu_num)) - (c_num1 - clu_num)/2
    c_num2 = len(cluster_dict2)
    likhood += c_num2*math.log(c_num2) - c_num2*math.log(len(cluster_dict)) - (c_num2/2)*math.log(2*math.pi)\
               - (c_num2*plu_num(cluster_dict2)/2)*math.log(calculate_sigma(cluster_dict2, ct2_tree, clu_num)) - (c_num2 - clu_num)/2
    return likhood

def BIC_before(cluster_dict, ct_tree, clu_num):
    bic = calculate_likelihood_before(cluster_dict, ct_tree, clu_num) - (clu_num*(plu_num(cluster_dict)+1)/2)*math.log(clu_num)
    return bic

def BIC_after(cluster_dict,cluster_dict1,ct1_tree,cluster_dict2,ct2_tree,clu_num):
    bic = calculate_likelihood_after(cluster_dict,cluster_dict1,ct1_tree,cluster_dict2,ct2_tree,clu_num) - (clu_num*(plu_num(cluster_dict)+1)/2)*math.log(clu_num)
    return bic

# In[93]:


def FTCAlgorithm(ftctree_dict):
    final_cluster = []
    final_centroids = []
    final_result = {}
    ct_tree = GetCT(ftctree_dict)
    print('-------------------------')
    print('Start Kmeans')
    try:
        result = Kmeans(ftctree_dict)
    except:
        print('不可分裂')
        final_cluster .append(list(ftctree_dict.keys()))
        final_centroids.append(ct_tree)
        final_result['cluster'] = final_cluster
        final_result['centroids'] = final_centroids
        return final_result
    print('Kmeans 结束')
    cluster = result['cluster']
    ct1 = list(cluster.keys())[0]
    ct2 = list(cluster.keys())[1]
    cluster1_dict = {}
    cluster2_dict = {}
    ct1_tree = result['centroids'][ct1]
    ct2_tree = result['centroids'][ct2]
    for item in cluster[ct1]:
        cluster1_dict[item] = ftctree_dict[item]
    for item in cluster[ct2]:
        cluster2_dict[item] = ftctree_dict[item]
    print('计算前后BIC')
    before_bic = BIC_before(ftctree_dict,ct_tree,1)
    after_bic = BIC_after(ftctree_dict,cluster1_dict,ct1_tree,cluster2_dict,ct2_tree,2)
    if before_bic < after_bic:
        print('可以分裂')
        temp_result1 = FTCAlgorithm(cluster1_dict)
        temp_result2 = FTCAlgorithm(cluster2_dict)
        for item in temp_result1['cluster']:
            final_cluster.append(item)
        for item in temp_result1['centroids']:
            final_centroids.append(item)
        for item in temp_result2['cluster']:
            final_cluster.append(item)
        for item in temp_result2['centroids']:
            final_centroids.append(item)
        final_result['cluster'] = final_cluster
        final_result['centroids'] = final_centroids
        return final_result
    else:
        print('不可分裂')
        final_cluster.append(list(ftctree_dict.keys()))
        final_centroids.append(ct_tree)
        final_result['cluster'] = final_cluster
        final_result['centroids'] = final_centroids
        return final_result


# 计算轮廓系数
def silhouette_score(ftctree_dict, cluster):
    sc = []
    # 遍历所有簇
    for index, item in enumerate(cluster):
        # 遍历簇中所有用户点
        for item2 in item:
            a = 0
            b_list = []
            b = 0
            # 对于每一个用户点，遍历簇中其他所有用户点
            for item3 in item:
                if item2 == item3:
                    continue
                t1 = ftctree_dict[item2]
                t2 = ftctree_dict[item3]
                a += Dist(inter_tree(t1,t2),union_tree(t1, t2))
            # 可能该簇中只有当前用户点一个点，此时a=0
            if a != 0:
                a = a/(len(item) - 1)

            # 遍历当前簇之外的其他簇
            for index4, item4 in enumerate(cluster):
                if index == index4:
                    continue
                # 遍历其他簇中所有用户点
                for item5 in item4:
                    t1 = ftctree_dict[item2]
                    t2 = ftctree_dict[item5]
                    b += Dist(inter_tree(t1, t2), union_tree(t1, t2))
                b = b/len(item4)
                b_list.append(b)
            b = min(b_list)
            sc.append((b-a)/max([a, b]))
    return mean(sc)


# 计算紧密性
def compactness(ftctree_dict, centroids, cluster):
    # 遍历所有簇
    for index, item in enumerate(cluster):
        # 每个簇中所有点到质心的距离列表
        cpi = []
        # 遍历当前簇中所有点
        t2 = centroids[index]
        for item2 in item:
            t1 = ftctree_dict[item2]
            cpi.append(Dist(inter_tree(t1, t2), union_tree(t1, t2)))
    return mean(cpi)

total_dict = deepcopy(ftctree_dict)
final_result = FTCAlgorithm(ftctree_dict)
print('-----------------------')
print('final_result')
print(final_result)
print('-----------------------')

print('-----------------------')
print('sc')
print(silhouette_score(total_dict, final_result['cluster']))
print('-----------------------')

print('-----------------------')
print('cp')
print(compactness(total_dict, final_result['centroids'], final_result['cluster']))
print('-----------------------')

for item in final_result['centroids']:
    item.show()




