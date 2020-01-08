# 通过entity2id文件，将外部KG中与entity有关的三元组提取出来
# 形成文件entity_content，先不做任何处理

import re
import os
import requests, json

data_dir = "data/HINdata"
entity_list = []
entity_dict = {}
triple_list = []

with open(os.path.join(data_dir, "entity2id.txt"), 'r', encoding='utf-8-sig') as f2:
    for line in f2:
        text = line.strip().split('\t')
        entity_list.append(text[0])
        entity_dict[text[0]] = text[1]

# print(entity_list)

for entity in entity_list:
    r = requests.get('https://api.ownthink.com/kg/knowledge?entity=%s' % (entity))
    data = json.loads(r.text).get("data")
    if len(data) != 0:
        desc = json.loads(r.text).get("data").get('desc')   # 字符串
        tag = json.loads(r.text).get("data").get('tag')     # tag列表
        atti = json.loads(r.text).get("data").get('avp')    # 属性列表
        tag_list = list(tag)
        atti_list = list(atti)
        if len(desc) != 0:
            w_str = entity_dict[entity] + '\t' + entity + '\t描述\t' + str(desc) + '\n'
            print(w_str)
            triple_list.append(w_str)
        if len(tag_list) != 0:
            w_str = entity_dict[entity] + '\t' + entity + '\t标签\t' + str(tag_list) + '\n'
            print(w_str)
            triple_list.append(w_str)
        if len(atti_list) != 0:
            w_str = entity_dict[entity] + '\t' + entity + '\t属性\t' + str(atti_list) + '\n'
            print(w_str)
            triple_list.append(w_str)

with open(os.path.join(data_dir, "entity_content_ownthink.txt"), 'w', encoding='utf-8-sig') as f4:
    for line in triple_list:
        f4.write(line)
