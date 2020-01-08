# 要有训练好的词向量模型才可以执行（运行word2vec_train.py）
# 挖掘实体之间的关系（难点）
import pynlpir
from gensim.models import KeyedVectors
from pyemd import emd
import os

data_dir = "data/HINdata"
# 停用词载入
stopwords = []
stopword = open("E:/BUAA/PyProject/stopwords-master/stopword.txt", 'rt', encoding='utf-8')
for line in stopword:
    stopwords.append(line.strip())

# 载入待测文本,使用ownthink知识库
text_id_list = []
tag_id_list = []
atti_id_list = []
with open(os.path.join(data_dir, "entity_content_ownthink.txt"), 'r', encoding='utf-8-sig') as f1:
    for line in f1:
        text = line.split('\t')
        if len(text) > 3:
            if text[2] == '描述':
                text_id_list.append([text[3].strip(), text[0].strip()])
            elif text[2] == '标签':
                tag_id_list.append([text[3].strip(), text[0].strip()])
            elif text[2] == '属性':
                atti_list = eval(text[3].strip())
                for atti in atti_list:
                    atti_id_list.append([atti[1], text[0].strip()])


# 载入实体集，构造字典<id,word>
entity_dict = {}
with open(os.path.join(data_dir, "entity2id.txt"), 'r', encoding='utf-8-sig') as f2:
    for line in f2:
        text = line.split('\t')
        entity_dict[text[1].strip()] = text[0]

# 待测文本分词、去停用词
sentence_id_list = []
pynlpir.open()
for desc in text_id_list:
    segments = pynlpir.segment(desc[0])
    words = []
    for segment in segments:
        words.append(segment[0])
    key = [",", "?", "[", "]", " ", "【", "】", "~"]
    words = [c for c in words if c not in key]
    words = [c for c in words if c not in stopwords]
    sentence_id_list.append([words, desc[1]])
pynlpir.close()

# 载入训练好的词向量模型
model = KeyedVectors.load_word2vec_format(os.path.join(data_dir, "Word2VecModel.txt"))
model.init_sims(replace=True)

with open(os.path.join(data_dir, "entity2entity.txt"), 'w', encoding='utf-8-sig') as f3:
    '''
    # 文本距离
    for i in range(len(sentence_id_list)-1):
        j = i + 1
        while j < len(sentence_id_list):
            entity1 = entity_dict[sentence_id_list[i][1]]
            entity2 = entity_dict[sentence_id_list[j][1]]
            distance = model.wmdistance(sentence_id_list[i][0], sentence_id_list[j][0])
            # 描述距离相近
            if distance > 0:
                f3.write(sentence_id_list[i][1] + '\t' + sentence_id_list[j][1] + '\n')
                print(entity1 + '\t' + entity2 + '\t' + str(distance))
            j += 1
    '''
    # 直接包含
    e2e = set()
    for entity_id in entity_dict.keys():
        for tag_id in tag_id_list:
            if entity_dict[entity_id] in tag_id[0]:
                e2e.add(tag_id[1] + '\t' + entity_id + '\n')
                # print(entity_dict[entity_id] + '\t' + tag_id[0] + '\t' + tag_id[1])
        for atti_id in atti_id_list:
            if entity_dict[entity_id] in atti_id[0]:
                e2e.add(atti_id[1] + '\t' + entity_id + '\n')
                # print(entity_dict[entity_id] + '\t' + atti_id[0] + '\t' + atti_id[1])
    for w_str in e2e:
        id = w_str.strip().split('\t')
        if id[0] != id[1]:
            f3.write(w_str)
        # print(w_str)
