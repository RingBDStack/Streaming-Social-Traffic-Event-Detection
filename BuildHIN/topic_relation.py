# 构建keyword2topic：直接判断keyword是否在topic的top-k特征中
# 构建keyword2keyword：近义词工具添加关键词之间的关系
import synonyms
import os

data_dir = "data/HINdata"
keyword_list = []
keyword_dict = {}
topic_list = []
word_synonyms_list = []

topic_n_words = 5

with open(os.path.join(data_dir, "keyword2id.txt"), 'r', encoding='utf-8-sig') as f1:
    for line in f1:
        text = line.split('\t')
        keyword_list.append(text[0])
        keyword_dict[text[0]] = text[1].strip()
        synonyms_list = synonyms.nearby(text[0])
        word_synonyms_list.append(synonyms_list[0])

with open(os.path.join(data_dir, "topic2id.txt"), 'r', encoding='utf-8-sig') as f2:
    for line in f2:
        text = line.split('\t')
        topic_word = text[0].split(', ')    # 逗号+空格
        topic_list.append([topic_word[0:topic_n_words], text[2].strip()])

with open(os.path.join(data_dir, "keyword2topic.txt"), 'w', encoding='utf-8-sig') as f3:
    for topic in topic_list:
        topic_word_list = topic[0]
        for topic_word in topic_word_list:
            if topic_word in keyword_list:
                f3.write(keyword_dict[topic_word]+'\t'+topic[1]+'\n')

with open(os.path.join(data_dir, "keyword2keyword.txt"), 'w', encoding='utf-8-sig') as f4:
    for i in range(len(keyword_list)-1):
        j = i + 1
        while j < len(keyword_list):
            if keyword_list[i] in word_synonyms_list[j]:
                f4.write(keyword_dict[keyword_list[i]] + '\t' + keyword_dict[keyword_list[j]] + '\n')
            j += 1
