# 构建三个文件：user2id、instance2id、user2instance
# user2user预存
import os
import pynlpir
from stanfordcorenlp import StanfordCoreNLP

oridata_dir = "data/oridata"
data_dir = "data/HINdata"

user_list = set()
instance_list = []
user2instance = []
with open(os.path.join(oridata_dir, "oridata.txt"), 'r', encoding='utf-8-sig') as f:
    for line in f:
        text = line.split('\t')
        user_list.add(text[0])
        instance_list.append(text[1].strip())
        user2instance.append([text[0], text[1].strip()])

instance_dict = {}
# 文本内容event instance到id的映射
with open(os.path.join(data_dir, "instance2id.txt"), 'w', encoding='utf-8-sig') as f1:
    for i in range(len(instance_list)):
        f1.write(instance_list[i] + '\ti' + str(i) + '\n')
        instance_dict[instance_list[i]] = 'i' + str(i)
print("instance2id is done.")

user_dict = {}
# 用户user到id的映射
with open(os.path.join(data_dir, "user2id.txt"), 'w', encoding='utf-8-sig') as f2:
    i = 0
    for user in user_list:
        f2.write(user + '\tu' + str(i) + '\n')
        user_dict[user] = 'u' + str(i)
        i += 1
print("user2id is done.")

with open(os.path.join(data_dir, "user2instance.txt"), 'w', encoding='utf-8-sig') as f3:
    for user2instance_list in user2instance:
        user2instance_str = user_dict[user2instance_list[0]] + '\t' + instance_dict[user2instance_list[1]] + '\n'
        f3.write(user2instance_str)
print("user2instance is done.")

# Step2：关键词提取，生成文件keyword2id、instance2keyword
keyword_list = set()
instance2keyword = []

# 利用NLPIR工具从instance_list内提取关键词
pynlpir.open()
for i in range(len(instance_list)):
    keyword_pair_list = pynlpir.get_key_words(instance_list[i], weighted=True)
    for keyword_pair in keyword_pair_list:
        keyword_list.add(keyword_pair[0])
        instance2keyword.append(['i'+str(i), keyword_pair[0]])
pynlpir.close()

keyword_dict = {}
with open(os.path.join(data_dir, "keyword2id.txt"),'w',encoding='utf-8-sig') as f_k2id:
    keyword_id = 0
    for keyword in keyword_list:
        f_k2id.write(keyword + '\tk' + str(keyword_id) + '\n')
        keyword_dict[keyword] = "k" + str(keyword_id)
        keyword_id += 1
print("keyword2id is done.")

with open(os.path.join(data_dir, "instance2keyword.txt"), 'w', encoding='utf-8-sig') as f_i2k:
    for instance2keyword_str_list in instance2keyword:
        instance2keyword_str = instance2keyword_str_list[0] + '\t' + keyword_dict[instance2keyword_str_list[1]] + '\n'
        f_i2k.write(instance2keyword_str)
print("instance2keyword is done.")

# Step3：利用CoreNLP从instance_list中提取实体entity
# entity部分生成两个文件：entity2id、instance2entity

instance2entity = []

'''
# 方法一（暂时弃用）：CoreNLP外部Java包，启动较慢，尽量所有工作一次启动搞定
nlp = StanfordCoreNLP(r'E:\stanford-corenlp-full-2018-10-05', lang='zh')
for i in range(len(instance_list)):
    # 提取实体代码
    entity_pair_list = nlp.ner(instance_list[i])    # 实体提取工具
    for entity_pair in entity_pair_list:
        if entity_pair[1] is not 'O':
            entity_list.add(entity_pair[0] + '\t' + entity_pair[1])
            instance2entity_str_list = ['i' + str(i), entity_pair[0] + '\t' + entity_pair[1]]
            instance2entity.append(instance2entity_str_list)
nlp.close()
'''
pynlpir.open()
raw_entities = []
with open("data\oridata\oridata.txt", 'r', encoding='utf-8-sig') as f:
    i = 0
    for line in f:
        text = line.strip().split('\t')
        segments = pynlpir.segment(text[1], pos_names = 'all')
        for segment in segments:
            try:
                if (segment[1] == 'noun:other proper noun')|(segment[1] == 'noun:organization/group name'):
                    raw_entities.append(segment[0])
                    instance2entity_str_list = ['i' + str(i), segment[0]]
                    instance2entity.append(instance2entity_str_list)
                elif segment[1].startswith('noun:personal name'):
                    raw_entities.append(segment[0])
                    instance2entity_str_list = ['i' + str(i), segment[0]]
                    instance2entity.append(instance2entity_str_list)
                elif segment[1].startswith('noun:toponym'):
                    raw_entities.append(segment[0])
                    instance2entity_str_list = ['i' + str(i), segment[0]]
                    instance2entity.append(instance2entity_str_list)
                elif (segment[1] == 'noun') and len(segment[0]) > 1:    # 个人添加的，有主观倾向
                    raw_entities.append(segment[0])
                    instance2entity_str_list = ['i' + str(i), segment[0]]
                    instance2entity.append(instance2entity_str_list)
            except:
                continue
        i += 1
        # print(str(raw_entities) + '\t' + str(raw_entities2))
pynlpir.close()

entity_list = list(set(raw_entities))

# 写文件entity2id，顺便构造字典
entity_dict = {}
with open(os.path.join(data_dir, "entity2id.txt"), 'w', encoding='utf-8-sig') as f_e2id:
    entity_id = 0
    for entity in entity_list:
        f_e2id.write(entity + '\t' + 'e' + str(entity_id) + '\n')
        entity_dict[entity] = "e" + str(entity_id)
        entity_id += 1
print("entity2id is done.")

with open(os.path.join(data_dir, "instance2entity.txt"), 'w', encoding='utf-8-sig') as f_i2e:
    for instance2entity_str_list in instance2entity:
        # print(str(instance2entity_str_list))
        instance2entity_str = instance2entity_str_list[0] + '\t' + entity_dict[instance2entity_str_list[1]] + '\n'
        f_i2e.write(instance2entity_str)
print("instance2entity is done.")





