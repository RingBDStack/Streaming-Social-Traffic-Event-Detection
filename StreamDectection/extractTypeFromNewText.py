import pynlpir
from gensim import corpora, models, similarities
import os

# 停用词载入
stopwords = []
stopword = open("E:/BUAA/PyProject/stopwords-master/哈工大停用词表.txt", 'rt', encoding='utf-8')
for line in stopword:
    stopwords.append(line.strip())
stop_char = [':', '：', '月', '日', '时', '分', '@']


def not_have_stop_char(word):
    flag = True
    for c in stop_char:
        if c in word:
            flag = False
            break
    return flag


def text2wordcut(text):
    segments = pynlpir.segment(text)
    words = []
    for segment in segments:
        words.append(segment[0])
    key = [",", "?", "[", "]", " ", "【", "】"]
    words = [c for c in words if c not in key]
    words = [c for c in words if c not in stopwords]
    words = [c for c in words if not_have_stop_char(c)]
    return words


def find_raw_entity(text, entity_dict):
    raw_entities = []

    segments = pynlpir.segment(text, pos_names = 'all')
    for segment in segments:
        try:
            if (segment[1] == 'noun:other proper noun') | (segment[1] == 'noun:organization/group name'):
                raw_entities.append(segment[0])
            elif segment[1].startswith('noun:personal name'):
                raw_entities.append(segment[0])
            elif segment[1].startswith('noun:toponym'):
                raw_entities.append(segment[0])
            elif (segment[1] == 'noun') and len(segment[0]) > 1:  # 个人添加的，有主观倾向
                raw_entities.append(segment[0])
        except:
            continue

    entity_list = list(set(raw_entities))
    entity_id_list = []
    for entity in entity_list:
        try:
            entity_id_list.append(entity_dict[entity])
        except:
            entity_id_list.append(entity)
    return entity_id_list


def find_keyword(text, keyword_dict):
    keyword_list = []
    keyword_pair_list = pynlpir.get_key_words(text, weighted=True)
    for keyword_pair in keyword_pair_list:
        keyword_list.append(keyword_pair[0])
    keyword_id_list = []
    for keyword in keyword_list:
        try:
            keyword_id_list.append(keyword_dict[keyword])
        except:
            keyword_id_list.append(keyword)
    return keyword_id_list


# 根据topic2id文件，返回所有叶节点上的主题关键词和topic_id
def get_topic_word_list(path):
    topic_word_list = []
    all_topic_list = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            text = line.strip().split('\t')
            if text[1] == 'l2':
                word_list = text[0].split(', ')
                id = text[2]
                topic_word_list.append([word_list[0:5], id])
                all_topic_list.append(word_list[0:5])
    return topic_word_list, all_topic_list


# 基于文本相似度返回预测的主题id
def find_topic(text, topic_word_list, all_topic_list):
    text_wordcut = text2wordcut(text)
    dictionary = corpora.Dictionary(all_topic_list)
    corpus = [dictionary.doc2bow(doc) for doc in all_topic_list]
    doc_test_vec = dictionary.doc2bow(text_wordcut)
    tfidf = models.TfidfModel(corpus)
    index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary.keys()))
    sim = index[tfidf[doc_test_vec]]
    sim_list = sim.tolist()
    max_index = sim_list.index(max(sim_list))
    # print(sim)
    topic_id = topic_word_list[max_index][1]
    return topic_id

'''
def txt2dict():
    user_dict = {}
    entity_dict = {}
    keyword_dict = {}
    with open(os.path.join(data_dir, "user2id.txt"), 'r', encoding='utf-8-sig') as f:
        for line in f:
            text = line.strip().split('\t')
            user_dict[text[0]] = text[1]
    with open(os.path.join(data_dir, "entity2id.txt"), 'r', encoding='utf-8-sig') as f:
        for line in f:
            text = line.strip().split('\t')
            entity_dict[text[0]] = text[2]
    with open(os.path.join(data_dir, "keyword2id.txt"), 'r', encoding='utf-8-sig') as f:
        for line in f:
            text = line.strip().split('\t')
            keyword_dict[text[0]] = text[1]
    return user_dict, entity_dict, keyword_dict
'''

def extract_text_type(user, text, user_dict, entity_dict, keyword_dict, topic_word_list, all_topic_list):
    pynlpir.open()
    user_id = user_dict[user]
    entity_list = find_raw_entity(text, entity_dict)
    keyword_list = find_keyword(text, keyword_dict)
    topic_id = find_topic(text, topic_word_list, all_topic_list)
    pynlpir.close()
    return user_id, entity_list, keyword_list, topic_id




