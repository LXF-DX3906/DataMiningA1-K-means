import pandas as pd
from copy import deepcopy
import json
from Kmeans2 import KMeans

class Method1:
    def __init__(self):
        self.str_file = '../customer_goods_info.json'
        self.customer_goods_info = None
        with open(self.str_file, 'r') as f:
            print("Load str file from {}".format(self.str_file))
            self.customer_goods_info = json.load(f)

    # 得到在n级品类结构汇总的所有商品pluno
    def get_pluno_list(self, n):
        df = pd.read_csv('../trade_new.csv')
        # 取出所需要的信息并根据pluno排序
        useful_info = df[['pluno', 'vipno', 'amt']].sort_values('pluno')
        # 取出所有的pluno列表（去重）
        pluno = useful_info.drop_duplicates(subset=['pluno'], keep='first', inplace=False)['pluno']
        pluno = [(int(item/10**(7-n))) for item in pluno]
        pluno = list(set(pluno))
        return pluno

    # 将所有商品在第n级品类结构汇总
    def n_classfication(self, n):
        data = {}
        for key in self.customer_goods_info:
            data[key] = {}
            data[key]['pluno'] = [int((int(item))/10**(7-n)) for item in self.customer_goods_info[key]['pluno']]
            data[key]['amt'] = [float(item) for item in self.customer_goods_info[key]['amt']]

        # last_data = {vipno:{pluno1:amt1, pluno2:amt2...}
        last_data = deepcopy(data)
        for index in data:
            temp_amt = 0
            temp_result = {}
            # temp_result['pluno'] = []
            # temp_result['amt'] = []
            current_pluno = data[index]['pluno'][0]
            for index2, item in enumerate(data[index]['pluno']):
                if current_pluno == item:
                    temp_amt += data[index]['amt'][index2]
                else:
                    temp_result[item] = temp_amt
                    current_pluno = item
                    temp_amt = data[index]['amt'][index2]
                if index2 == len(data[index]['pluno']) - 1:
                    temp_result[item] = temp_amt
                    break
            last_data[index] = temp_result

        return last_data

    # 计算jaccard值
    def cpmpute_jaccard(self, c1, c2):

        same_pluno_list = [x for x in c1.keys() if x in c2.keys()]
        diff_pluno_list = [y for y in (list(c1.keys()) + list(c2.keys())) if y not in same_pluno_list]

        same_list_max = []
        same_list_min = []
        different_list = []

        for item in same_pluno_list:
            if c1[item] >= c2[item]:
                same_list_max.append(c1[item])
                same_list_min.append(c2[item])
            else:
                same_list_max.append(c2[item])
                same_list_min.append(c1[item])
        for item in diff_pluno_list:
            if item in c1.keys():
                different_list.append(c1[item])
            else:
                different_list.append(c2[item])
        return sum(same_list_min) / (sum(same_list_max, sum(different_list)))

    # 得到所有jaccard值
    def get_jaccard(self, lev_n_classification_info):
        jaccard_sim = {}
        for index1 in lev_n_classification_info:
            for index2 in list(lev_n_classification_info.keys())[list(lev_n_classification_info.keys()).index(index1):]:
                if index1 == index2:
                    continue
                c1 = lev_n_classification_info[index1]
                c2 = lev_n_classification_info[index2]
                jaccard_sim[index1+'|%|'+index2] = self.cpmpute_jaccard(c1, c2)
        return jaccard_sim

    def coordinate(self, pluno_list, lev_n_classification_info):
        codi = {}
        for index in lev_n_classification_info:
            codi[index] = []
            for item in pluno_list:
                if item not in list(lev_n_classification_info[index].keys()):
                    codi[index].append(0)
                else:
                    codi[index].append(lev_n_classification_info[index][item])
        return codi

    def main(self):
        classification_info = {}
        pluno_list = {}
        # 1到4级品类结构
        for i in range(1, 5):
            classification_info[i] = self.n_classfication(i)
            pluno_list[i] = self.get_pluno_list(i)

        jaccard_sim_sum = {}
        jaccard_sim_avg = {}
        i = 0
        # 求平均jaccard
        for index in classification_info:
            if i == 0:
                jaccard_sim_sum = self.get_jaccard(classification_info[index])
            jaccard_sim = self.get_jaccard(classification_info[index])
            for index2 in jaccard_sim_sum:
                jaccard_sim_sum[index2] += jaccard_sim[index2]
            i += 1
        for index2 in jaccard_sim_sum:
            jaccard_sim_avg[index2] = jaccard_sim_sum[index2]/len(list(classification_info.keys()))


        # 求各个用户点坐标
        codi = {}
        for index in classification_info:
            codi[index] = {}
            codi[index] = self.coordinate(pluno_list[index], classification_info[index])

        sc_list = {}
        i = 20
        while i <= 30:
            print("---------------------")
            print("n_clusters: " + str(i))
            kmeans = KMeans(i, 50, codi, jaccard_sim_avg)
            result = kmeans.kMeans()
            sc = kmeans.silhouette_score(result['cluster'])
            cp = kmeans.compactness(result['cluster'])
            print("sc: " + str(sc))
            print("cp: " + str(cp))
            print("---------------------")
            sc_list[i] = sc
            i += 1


# with open("fourth_classification.json", "w") as f:
#     f.write(json.dumps(last_data, ensure_ascii=False, indent=4, separators=(',', ':')))

if __name__ == "__main__":
    method1 = Method1()
    method1.main()

    print(1)
