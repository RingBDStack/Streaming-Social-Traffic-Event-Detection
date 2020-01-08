# 必须先生成文件entity_content才能执行
# 提取关键词在实体描述中的从属关系

import os

data_dir = "data/HINdata"
keyword_list = []
keyword_dict = {}

with open(os.path.join(data_dir, "keyword2id.txt"), 'r', encoding='utf-8-sig') as f1:
    for line in f1:
        text = line.split('\t')
        keyword_list.append(text[0])
        keyword_dict[text[0]] = text[1].strip()

keyword2entity = set()
with open(os.path.join(data_dir, "entity_content_ownthink.txt"), 'r', encoding='utf-8-sig') as f2:
    for line in f2:
        text = line.strip().split('\t')
        if len(text) > 3:
            for keyword in keyword_list:
                if keyword in text[3]:
                    keyword2entity.add(keyword_dict[keyword] + '\t' + text[0] + '\n')

with open(os.path.join(data_dir, "keyword2entity_ownthink.txt"), 'w', encoding='utf-8-sig') as f3:
    for line in keyword2entity:
        f3.write(line)