import pandas as pd
from copy import deepcopy
import json

class Method1:
    def __init__(self):
        self.str_file = './customer_goods_info.json'
        self.customer_goods_info = None
        with open(self.str_file, 'r') as f:
            print("Load str file from {}".format(self.str_file))
            self.customer_goods_info = json.load(f)

    # 将所有商品在第四级品类结构汇总
    def fourth_classfication(self):
        data = {}
        for key in self.customer_goods_info:
            data[key] = {}
            data[key]['pluno'] = [int((int(item))/1000) for item in self.customer_goods_info[key]['pluno']]
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
    def compute_jaccard(self, fourth_classification_info):
        jaccard_sim = {}
        for index1 in fourth_classification_info:
            for index2 in list(fourth_classification_info.keys())[list(fourth_classification_info.keys()).index(index1):]:

                if index1 == index2:
                    continue
                c1 = fourth_classification_info[index1]
                c2 = fourth_classification_info[index2]

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
                jaccard_sim[index1+'|%|'+index2] = sum(same_list_min)/(sum(same_list_max, sum(different_list)))
        return jaccard_sim
# with open("fourth_classification.json", "w") as f:
#     f.write(json.dumps(last_data, ensure_ascii=False, indent=4, separators=(',', ':')))

if __name__ == "__main__":
    method1 = Method1()
    fourth_classification_info = method1.fourth_classfication()
    jaccard_sim = method1.compute_jaccard(fourth_classification_info)
    print(jaccard_sim)
