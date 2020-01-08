# 利用HLDA生成层次主题模型
# 输出三个文件：topic2id、topic2topic、instance2topic

from hlda.sampler import HierarchicalLDA
import os
import _pickle as cPickle
import gzip

oridata_dir = "data/oridata"
data_dir = "data/HINdata"
corpus = []     # 语料库，结构为文本词列表的列表（两层）
vocab = set()   # 无序不重复全体单词集合

# stop_pos = ['PU', 'DEC', 'DEG', 'DER', 'DEV', 'AS', 'SP', 'MSP', 'BA', 'VV', 'CC', 'CS', 'P', 'PN', 'AD', 'M']
# high_fre_word = ['交通', '方向', '路段']

with open("data/oridata/wordcut_text.txt", 'r', encoding='utf-8-sig') as f:
    for line in f:
        sub_words_ls = line.strip().split(' ')
        corpus.append(sub_words_ls)
        vocab.update(sub_words_ls)

# 生成vocab的倒排索引
vocab = sorted(list(vocab))
vocab_index = {}
for i, w in enumerate(vocab):
    vocab_index[w] = i

print(len(vocab))

# 将文本中的词更换为索引
new_corpus = []
for doc in corpus:
    new_doc = []
    for word in doc:
        word_idx = vocab_index[word]
        new_doc.append(word_idx)
    new_corpus.append(new_doc)
print(len(new_corpus))

# HLDA参数设置
n_samples = 500       # no of iterations for the sampler 迭代次数
alpha = 10.0          # smoothing over level distributions
gamma = 1.0           # CRP smoothing parameter; number of imaginary customers at next, as yet unused table
eta = 0.1             # smoothing over topic-word distributions
num_levels = 3        # the number of levels in the tree 主题层数
display_topics = 50   # the number of iterations between printing a brief summary of the topics so far 迭代显示步长
n_words = 5           # the number of most probable words to print for each topic after model estimation 主题关键词显示数
with_weights = False  # whether to print the words with the weights 是否显示权重

hlda = HierarchicalLDA(new_corpus, vocab, alpha=alpha, gamma=gamma, eta=eta, num_levels=num_levels)
hlda.estimate(n_samples, display_topics=display_topics, n_words=n_words, with_weights=with_weights)

# 打印结果
topic2id = set()
instance2topic = []
topic2topic = set()

for d in range(len(corpus)):
    node = hlda.document_leaves[d]
    path = []
    while node is not None:
        path.append(node)
        node = node.parent
    path.reverse()  # 根到叶节点的路径

    # output_line = all_docs[d]
    instance_id = 'i' + str(d)

    n_words = 5
    with_weights = False
    topic_id = []
    level_id = []
    for n in range(len(path)):
        node = path[n]
        topic2id_str = node.get_top_words(n_words, with_weights) + '\t'
        topic2id_str += 'l' + str(node.level) + '\tt' + str(node.node_id)
        level_id.append(node.level)
        topic_id.append(node.node_id)
        topic2id.add(topic2id_str)
    for n in range(len(topic_id)-1):
        parent_id = topic_id[n]
        chile_id = topic_id[n+1]
        topic2topic_str = 't' + str(parent_id) + '\tt' + str(chile_id)
        topic2topic.add(topic2topic_str)
    instance2topic_str = instance_id + '\tt' + str(topic_id[len(topic_id)-1])
    instance2topic.append(instance2topic_str)
    # print(output_line)

tid_dict = {}
tid_new = 0
tmp_list = []
# 整理topic的序号
for line in topic2id:
    tmp = line.split('\t')
    tid_old = tmp[2]
    tmp_list.append(tid_old)
tmp_list.sort()
for tmp in tmp_list:
    tid_dict[tmp] = "t" + str(tid_new)
    tid_new += 1
print(tid_dict)

# 写入文件
with open(os.path.join(data_dir, 'topic2id.txt'), 'w', encoding='utf-8-sig') as f1:
    for line in topic2id:
        text = line.split('\t')
        f1.write(text[0]+'\t'+text[1]+'\t'+tid_dict[text[2]]+'\n')
with open(os.path.join(data_dir, 'instance2topic.txt'), 'w', encoding='utf-8-sig') as f2:
    for line in instance2topic:
        text = line.split('\t')
        f2.write(text[0] + '\t' + tid_dict[text[1]] + '\n')
with open(os.path.join(data_dir, 'topic2topic.txt'), 'w', encoding='utf-8-sig') as f3:
    for line in topic2topic:
        text = line.split('\t')
        f3.write(tid_dict[text[0]] + '\t' + tid_dict[text[1]] + '\n')


def save_zipped_pickle(obj, filename, protocol=-1):
    with gzip.open(filename, 'wb') as f:
        cPickle.dump(obj, f, protocol)


def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = cPickle.load(f)
        return loaded_object


save_zipped_pickle(hlda, os.path.join(data_dir, 'topic_hlda.p'))
