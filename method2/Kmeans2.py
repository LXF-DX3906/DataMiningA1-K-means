# 1 质心个数，质心的选取（两个一个为centroids，另一个为last_centroids）
# 2 根据质心计算簇
# 3 得到簇后，计算簇中心
# 4 判断簇是否变化，如变化则继续（更新last_centroids），不变则退出
# 5 簇中心更新后，计算簇中心到其他所有点的jaccard距离，并保存，跳转至步骤2
import random
import pandas as pd
import operator
from copy import deepcopy
from copy import copy
from numpy import mean


class KMeans:
    def __init__(self, n_cluster, max_iter, codi, jaccard_sim):
        self.n_clusters = n_cluster
        self.max_iter = max_iter
        self.codi = codi
        self.jaccard_sim = jaccard_sim
        # { c1|%|p1: jaccard1, ...}
        self.centroids_jaccard_sim = {}
        # { lev_1: {c1:codi1, ...}}
        self.centroids = {}

    # 初始化质心
    def init_centroids(self):
        # 随机选取质心
        t_c_list = {}
        i = 0
        for index in self.codi:
            if i == 0:
                t_c_list = random.sample(list(self.codi[index].keys()), self.n_clusters)
            else:
                break
            i += 1

        # 初始化质心坐标
        for index in self.codi:
            self.centroids[index] = {}
            for item in t_c_list:
                self.centroids[index][item] = self.codi[index][item]

        # 初始化质心到各个点的jaccard距离
        # 遍历质心
        i = 0
        for lev_n in self.centroids:
            if i == 0:
                # 遍历质心
                for item in self.centroids[lev_n]:
                    # 遍历所有用户
                    for item2 in self.codi[lev_n]:
                        # 质心点等于用户点，跳过
                        if item == item2:
                            continue
                        c_name = str(item) + "|%|" + str(item2)
                        t_key1 = str(item) + "|%|" + str(item2)
                        t_key2 = str(item2) + "|%|" + str(item)
                        try:
                            self.centroids_jaccard_sim[c_name] = self.jaccard_sim[t_key1]
                        except:
                            self.centroids_jaccard_sim[c_name] = self.jaccard_sim[t_key2]
            else:
                break
            i += 1

    # 计算所有簇
    def compute_cluster(self):
        # 初始化簇字典
        # cluster: { centroid1: [c1_vipno1, c1_vipno2, ...], centroid2: [c2_vipno1, c2_vipno2, ...], ... }
        cluster = {}
        i = 0
        for lev_n in self.centroids:
            if i == 0:
                for index in self.centroids[lev_n]:
                    cluster[index] = []
            else:
                break
            i += 1
        # 遍历所有用户坐标
        i = 0
        for lev_n in self.centroids:
            if i == 0:
                for index in self.codi[lev_n]:
                    # 当前用户为质心（只在第一次迭代时会出现此种情况），将当前用户加入到对应簇中，跳过
                    if index in list(self.centroids[lev_n].keys()):
                        if 'centroid' not in index:
                            cluster[index].append(index)
                        continue
                    # 初始化用户点到质心距离字典
                    n_distance = {}
                    # 遍历所有质心
                    for index2 in self.centroids[lev_n]:
                        c_name = str(index2) + "|%|" + str(index)
                        # 用户点到质心的距离
                        n_distance[index2] = 1 - self.centroids_jaccard_sim[c_name]
                    min_c = min(n_distance, key=lambda x: n_distance[x])
                    cluster[min_c].append(index)
            else:
                break
            i += 1
        return cluster

    # 计算jaccard值
    def compute_jaccard(self, c1, c2):

        same_list_max = []
        same_list_min = []
        different_list = []
        same_list_max.append(0)
        same_list_min.append(0)
        different_list.append(0)

        for index, item in enumerate(c1):
            if item != 0 and c2[index] != 0:
                if item >= c2[index]:
                    same_list_max.append(item)
                    same_list_min.append(c2[index])
                else:
                    same_list_max.append(c2[index])
                    same_list_min.append(item)
            elif item != 0 and c2[index] == 0:
                different_list.append(item)
            elif item == 0 and c2[index] != 0:
                different_list.append(c2[index])

        return sum(same_list_min) / (sum(same_list_max, sum(different_list)))

    # 更新质心,并计算更新后的质心与其他点的jaccard值(质心命名：n + centroid + m（n：第n次计算，m：第m个质心）)
    def update_centroids(self, cluster, n):
        # m用来记录是第几个质心
        m = 0
        # 清空质心字典
        self.centroids = {}
        for lev_n in self.codi:
            self.centroids[lev_n] = {}
        # 清空质心与用户点的jaccard字典
        self.centroids_jaccard_sim = {}
        # 遍历每一个簇
        for index in cluster:
            m += 1
            # sum_codi用来记录当前簇中所有用户在每个维度上的值之和
            jaccard_sim_sum = {}
            i = 0
            for lev_n in self.codi:
                sum_codi = copy(self.codi[lev_n][cluster[index][0]])
                # 遍历当前簇中所有用户点
                for i2, item in enumerate(cluster[index]):
                    if i2 == 0:
                        continue
                    # 遍历sum_codi中的每一个维度，在每个维度上求和
                    for s_i, s_item in enumerate(sum_codi):
                        sum_codi[s_i] += self.codi[lev_n][item][s_i]

                # avg为质心坐标
                avg = []
                n_p = len(cluster[index])
                for s_item in sum_codi:
                    avg.append(s_item/n_p)

                c_name = str(n) + "centroid" + str(m)
                self.centroids[lev_n][c_name] = avg
                for index2 in self.codi[lev_n]:
                    t_name = c_name + "|%|" + str(index2)
                    if i == 0:
                        jaccard_sim_sum[t_name] = self.compute_jaccard(self.codi[lev_n][index2], avg)
                    else:
                        jaccard_sim_sum[t_name] += self.compute_jaccard(self.codi[lev_n][index2], avg)
                i += 1
            for index2 in jaccard_sim_sum:
                self.centroids_jaccard_sim[index2] = jaccard_sim_sum[index2] / len(list(self.codi.keys()))

    # 判断簇是否发生变化
    def is_change(self, last_cluster, cluster):
        equ_n = 0
        for index in cluster:
            for index2 in last_cluster:
                if operator.eq(cluster[index], last_cluster[index2]):
                    equ_n += 1
        if equ_n == self.n_clusters:
            return False
        return True

    # 计算轮廓系数
    def silhouette_score(self, cluster):
        sc = []
        # 遍历所有簇
        for index in cluster:
            # 遍历簇中所有用户点
            for index2, item2 in enumerate(cluster[index]):
                a = 0
                b_list = []
                b = 0
                # 对于每一个用户点，遍历簇中其他所有用户点
                for index3, item3 in enumerate(cluster[index]):
                    if index2 == index3:
                        continue
                    t_key1 = str(item2) + "|%|" + str(item3)
                    t_key2 = str(item3) + "|%|" + str(item2)
                    try:
                        a += 1 - self.jaccard_sim[t_key1]
                    except:
                        a += 1 - self.jaccard_sim[t_key2]
                # 可能该簇中只有当前用户点一个点，此时a=0
                if a != 0:
                    a = a/(len(cluster[index]) - 1)

                # 遍历当前簇之外的其他簇
                for index4 in cluster:
                    if index == index4:
                        continue
                    # 遍历其他簇中所有用户点
                    for item4 in cluster[index4]:
                        t_key1 = str(item2) + "|%|" + str(item4)
                        t_key2 = str(item4) + "|%|" + str(item2)
                        try:
                            b += 1 - self.jaccard_sim[t_key1]
                        except:
                            b += 1 - self.jaccard_sim[t_key2]
                    b = b/len(cluster[index4])
                    b_list.append(b)
                b = min(b_list)
                sc.append((b-a)/max([a, b]))
        return mean(sc)

    # 计算紧密性
    def compactness(self, cluster):
        # 遍历所有簇
        for index in cluster:
            # 每个簇中所有点到质心的距离列表
            cpi = []
            # 遍历当前簇中所有点
            for item in cluster[index]:
                t_name = str(index) + "|%|" + str(item)
                cpi.append(1 - self.centroids_jaccard_sim[t_name])
        return mean(cpi)


    def kMeans(self):
        self.init_centroids()
        i = 1
        while i <= self.max_iter:
            cluster = self.compute_cluster()
            is_change = True
            if i > 1:
                is_change = self.is_change(last_cluster, cluster)
            if not is_change:
                result = {}
                result['centroids'] = self.centroids
                result['centroids_jaccard_sim'] = self.centroids_jaccard_sim
                result['cluster'] = cluster

                return result

            last_cluster = deepcopy(cluster)
            self.update_centroids(cluster, i)

            # print("iter: " + str(i))
            i += 1

        result = {}
        result['centroids'] = self.centroids
        result['centroids_jaccard_sim'] = self.centroids_jaccard_sim
        result['cluster'] = cluster
        return result