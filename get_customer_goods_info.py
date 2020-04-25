# 数据预处理，取出所有客户以及所购买的商品编号与价格（同一商品被多次购买需将金额相加）
import pandas as pd
import json
df = pd.read_csv('trade_new.csv')

# 取出所需要的信息并根据vipno排序
useful_info = df[['pluno', 'vipno', 'amt']].sort_values('vipno')

# 取出所有的vipno列表（去重）
vipno = useful_info.drop_duplicates(subset=['vipno'], keep='first', inplace=False)['vipno']

# 以每个vipno为key，值为[(pluno, amt)]
data = {}
for item in vipno:
    data[item] = useful_info.loc[useful_info['vipno'] == item][['pluno', 'amt']]

# 由于一个商品可以被多次购买，所以对多次购买的商品的金额进行求和
for index in data:
    temp_data = data[index].sort_values('pluno')
    temp_pluno = temp_data.drop_duplicates(subset=['pluno'], keep='first', inplace=False)['pluno']
    temp_result = {}
    temp_result['pluno'] = []
    temp_result['amt'] = []
    for item in temp_pluno:
        t = temp_data.loc[temp_data['pluno'] == item]['amt'].sum()
        # temp_result['pluno'].append(int(item/1000))
        temp_result['pluno'].append(str(item))
        temp_result['amt'].append(str(t))
    data[index] = temp_result

with open("customer_goods_info.json", "w") as f:
    f.write(json.dumps(data, ensure_ascii=False, indent=4, separators=(',', ':')))
