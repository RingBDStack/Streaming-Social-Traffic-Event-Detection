import json
import requests
from extractTypeFromNewText import *
from init_dbscan import *
import numpy as np
import os

type_list = ['i', 'e', 'k', 'u', 't']  # 五个type：事件实例、实体、关键词、用户、主题
path_matrix_str_list = ['uu', 'ee', 'kk', 'tt', 'ui', 'ie', 'ik', 'it', 'ke', 'kt']
metapath_list = ['iui', 'iei', 'iki', 'iti', 'iuui', 'ieei', 'ikki', 'itti', 'iekei', 'ikeki', 'iktki', 'itkti',
                 'iekkei', 'ikeeki', 'ikttki', 'itkkti', 'ieekeei', 'ikkekki', 'ikktkki', 'ittktti', 'iektkei',
                 'itkekti']  # 22个，可扩充
den_w = len(metapath_list)

ori_dir = "data"
HIN_dir = "data/eventHIN"    # 数据文件根目录
GCN_dir = "data/GCNData"

weight_file = "data/results_new_22.txt"

orifile_name = "oridata.txt"
labelfile_name = "instance2label_content.txt"
userfile_name = "user2id.txt"
entityfile_name = "entity2id.txt"
keywordfile_name = "keyword2id.txt"
topicfile_name = "topic2id.txt"

type_num_dict = {}      # 各个type的数量
edge_matrix = {}        # 存放各超边对应的邻接矩阵

# dbscan参数
eps = 8
MinPts = 2

node_size = 32
s_index = 0


# 初始化已有的user, entity，keyword，topic，返回四个字典，值为key，id为value
def init_dict(root, userpath, entitypath, keywordpath, topicpath):
    user_dict = {}
    entity_dict = {}
    keyword_dict = {}
    topic_dict = {}
    with open(os.path.join(root, userpath), 'r', encoding='utf-8-sig') as f:
        for line in f:
            text = line.strip().split('\t')
            user_dict[text[0]] = text[1]
    with open(os.path.join(root, entitypath), 'r', encoding='utf-8-sig') as f:
        for line in f:
            text = line.strip().split('\t')
            entity_dict[text[0]] = text[1]
    with open(os.path.join(root, keywordpath), 'r', encoding='utf-8-sig') as f:
        for line in f:
            text = line.strip().split('\t')
            keyword_dict[text[0]] = text[1]
    # topic2id多一个level字段
    with open(os.path.join(root, topicpath), 'r', encoding='utf-8-sig') as f:
        for line in f:
            text = line.strip().split('\t')
            if text[1] == 'l2':
                topic_dict[text[0]] = text[2]

    return user_dict, entity_dict, keyword_dict, topic_dict


def init_HIN(root, oridata_file):
    new_text_list = []
    new_user_list = []
    # 原始数据文件：“用户\t文本内容”
    with open(os.path.join(root, oridata_file), 'r', encoding='utf-8-sig') as f:
        for line in f:
            text = line.strip().split('\t')
            new_user_list.append(text[0])
            new_text_list.append(text[1])

    # 获取type值和id的对应关系
    u_d, e_d, k_d, t_d = init_dict(HIN_dir, userfile_name, entityfile_name, keywordfile_name, topicfile_name)
    # 获取topic预测所需的语料库
    topic_word_list, all_topic_list = get_topic_word_list(os.path.join(HIN_dir, topicfile_name))
    # 设置各个邻接矩阵的大小
    type_num_dict['e'] = len(e_d)
    type_num_dict['k'] = len(k_d)
    type_num_dict['u'] = len(u_d)
    type_num_dict['i'] = len(new_text_list)
    topic_num = 0
    with open(os.path.join(HIN_dir, topicfile_name), 'r', encoding='utf-8-sig') as f:
        for line in f:
            topic_num += 1
    type_num_dict['t'] = topic_num


def generate_edge_matrix_from_file(sub_path, type1, type2):  # 生成两个type之间的邻接矩阵（01，不对称）
    matrix_w = np.zeros((type_num_dict[type1], type_num_dict[type2]))
    mat_index_list = []
    with open(os.path.join(HIN_dir, sub_path), 'r', encoding='utf-8-sig') as f:
        for line in f:
            ids = line.split('\t')
            id1 = ids[0].strip().replace(type1, '')
            id2 = ids[1].strip().replace(type2, '')
            mat_index_list.append([id1, id2])
    for index in mat_index_list:
        # print(index)
        matrix_w[int(index[0])][int(index[1])] = 1
    if type1 == type2:  # 自环边生成对称矩阵
        for index in mat_index_list:
            matrix_w[int(index[1])][int(index[0])] = 1
    return matrix_w


def generateAllEdgeMatrix():  # 生成所有元路径超边的矩阵
    edge_matrix["ee"] = generate_edge_matrix_from_file("entity2entity.txt", 'e', 'e')
    edge_matrix["ie"] = generate_edge_matrix_from_file("instance2entity.txt", 'i', 'e')
    edge_matrix["ik"] = generate_edge_matrix_from_file("instance2keyword.txt", 'i', 'k')
    edge_matrix["it"] = generate_edge_matrix_from_file("instance2topic.txt", 'i', 't')
    edge_matrix["ke"] = generate_edge_matrix_from_file("keyword2entity_ownthink.txt", 'k', 'e')
    edge_matrix["kk"] = generate_edge_matrix_from_file("keyword2keyword.txt", 'k', 'k')
    edge_matrix["kt"] = generate_edge_matrix_from_file("keyword2topic.txt", 'k', 't')
    edge_matrix["tt"] = generate_edge_matrix_from_file("topic2topic.txt", 't', 't')
    edge_matrix["ui"] = generate_edge_matrix_from_file("user2instance.txt", 'u', 'i')
    edge_matrix["uu"] = generate_edge_matrix_from_file("user2user.txt", 'u', 'u')
    # 非对称矩阵生成其转置矩阵
    edge_matrix["ei"] = edge_matrix["ie"].T
    edge_matrix["ki"] = edge_matrix["ik"].T
    edge_matrix["ti"] = edge_matrix["it"].T
    edge_matrix["ek"] = edge_matrix["ke"].T
    edge_matrix["tk"] = edge_matrix["kt"].T
    edge_matrix["iu"] = edge_matrix["ui"].T


def generate_Metapath_adjM(metapath):
    edge_list = []
    for i in range(len(metapath)-1):
        edge = metapath[i:i+2]
        edge_list.append(edge)
    size = type_num_dict['i']
    mul = np.eye(size, size, dtype='float32')
    for edge in edge_list:
        # print(str(edge)+" "+str(mul.shape)+' '+str(edge_matrix[edge].shape))
        mul = np.matmul(mul, edge_matrix[edge])
    return mul


def know_sim(M, i, j):
    broadness = M[i][i] + M[j][j]
    overlap = 2*M[i][j]
    if broadness == 0:
        return 0
    else:
        return overlap/broadness


def cal_KIES(w, i, j, metapath_matrix):
    sum = 0
    for k in range(len(metapath_list)):
        sum += w[k] * know_sim(metapath_matrix[k], i, j)
    return sum


# 根据instance下标和聚类结果获取index所在聚类的size
def get_cluster_size(index, result):
    target_cluster = result[index]
    count = 0
    for key in result.keys():
        if result[key] == target_cluster:
            count += 1
    return count


def add_newkeyword_2topic(keyword, id, topic_word_list):
    for topic in topic_word_list:
        if keyword in topic[0]:
            edge_matrix['kt'][id][int(topic[1].replace('t', ''))] = 1


def add_newkeyword_2entity(keyword, id, e_d):
    for key in e_d.keys():
        r = requests.get('https://api.ownthink.com/kg/knowledge?entity=%s' % (key))
        data = json.loads(r.text).get("data")
        if keyword in data:
            edge_matrix['ke'][id][int(e_d[key].replace('e', ''))] = 1


def add_newentity_fromkeyword(entity, id, k_d):
    r = requests.get('https://api.ownthink.com/kg/knowledge?entity=%s' % (entity))
    data = json.loads(r.text).get("data")
    for key in k_d.keys():
        if key in data:
            edge_matrix['ke'][int(k_d[key].replace('k', ''))][id] = 1


# 删除一个点，用于判断删掉这个点会否产生噪音点，以更改ptses
def delete_old_node(index, ptses, result):
    if ptses[index] == 2:
        cluster_size = get_cluster_size(index, result)
        if cluster_size == 2:
            for key in result.keys():
                if result[key] == result[index] and key != index:
                    ptses[key] = 0

    # 将其原来的行或列置0
    edge_matrix['iu'][index] = np.zeros(edge_matrix['iu'][index].shape)
    edge_matrix['ie'][index] = np.zeros(edge_matrix['ie'][index].shape)
    edge_matrix['ik'][index] = np.zeros(edge_matrix['ik'][index].shape)
    edge_matrix['it'][index] = np.zeros(edge_matrix['it'][index].shape)
    edge_matrix['ui'][:, index] = np.zeros(edge_matrix['ui'][:, index].shape)
    edge_matrix['ei'][:, index] = np.zeros(edge_matrix['ei'][:, index].shape)
    edge_matrix['ki'][:, index] = np.zeros(edge_matrix['ki'][:, index].shape)
    edge_matrix['ti'][:, index] = np.zeros(edge_matrix['ti'][:, index].shape)


# 更改edge_matrix与i相关的01矩阵
def change_edge_matrix(index, user, entity_list, keyword_list, topic, e_d, k_d, topic_word_list):
    user_id = int(user.strip().replace('u', ''))
    edge_matrix['ui'][user_id][index] = 1
    edge_matrix['iu'][index][user_id] = 1

    topic_id = int(topic.strip().replace('t', ''))
    edge_matrix['ti'][topic_id][index] = 1
    edge_matrix['it'][index][topic_id] = 1

    # entity与keyword有新增的可能，如果有新词，则扩充原始矩阵，并增加关系
    # entity的关系：ie, ee, ke
    for entity in entity_list:
        if 'e' in entity:
            entity_id = int(entity.strip().replace('e', ''))
            edge_matrix['ei'][entity_id][index] = 1
            edge_matrix['ie'][index][entity_id] = 1
        else:
            print("New Entity！")
            # 获取最大的entity_id
            max_id = type_num_dict['e']
            # 更改main内全局entity_dict
            e_d[entity] = 'e' + str(max_id)
            # 更改entity的全局数量
            type_num_dict['e'] += 1

            # ie, ee, ke三个矩阵扩容
            entity_id = int(max_id)
            # ie和ei:
            edge_matrix['ie'] = np.column_stack((edge_matrix['ie'], np.zeros((edge_matrix['ie'].shape[0], 1))))
            edge_matrix['ie'][index][entity_id] = 1
            edge_matrix['ei'] = edge_matrix['ie'].T
            # ke和ek:
            edge_matrix['ke'] = np.column_stack((edge_matrix['ke'], np.zeros((edge_matrix['ke'].shape[0], 1))))
            add_newentity_fromkeyword(entity, entity_id, k_d)
            edge_matrix['ek'] = edge_matrix['ke'].T
            # ee:
            edge_matrix['ee'] = np.column_stack((edge_matrix['ee'], np.zeros((edge_matrix['ee'].shape[0], 1))))
            edge_matrix['ee'] = np.row_stack((edge_matrix['ee'], np.zeros((1, edge_matrix['ee'].shape[1]))))
                # 暂时只扩容，关系尚未添加

    # keyword的关系：ik, kk, kt, ke
    for keyword in keyword_list:
        if 'k' in keyword:
            keyword_id = int(keyword.strip().replace('k', ''))
            edge_matrix['ki'][keyword_id][index] = 1
            edge_matrix['ik'][index][keyword_id] = 1
        else:
            print("New Keyword！")
            # 获取最大的entity_id
            max_id = type_num_dict['k']
            # 更改main内全局keyword_dict
            k_d[keyword] = 'k' + str(max_id)
            # 更改entity的全局数量
            type_num_dict['k'] += 1

            # ik, kk, kt, ke四个矩阵扩容
            keyword_id = int(max_id)
            # ik和ki:
            edge_matrix['ik'] = np.column_stack((edge_matrix['ik'], np.zeros((edge_matrix['ik'].shape[0], 1))))
            edge_matrix['ik'][index][keyword_id] = 1
            edge_matrix['ki'] = edge_matrix['ik'].T
            # ke和ek:
            edge_matrix['ke'] = np.row_stack((edge_matrix['ke'], np.zeros((1, edge_matrix['ke'].shape[1]))))
            # add_newkeyword_2entity(keyword, keyword_id, e_d) 耗时太长
            edge_matrix['ek'] = edge_matrix['ke'].T
            # kt和tk:
            edge_matrix['kt'] = np.row_stack((edge_matrix['kt'], np.zeros((1, edge_matrix['kt'].shape[1]))))
            add_newkeyword_2topic(keyword, keyword_id, topic_word_list)
            edge_matrix['tk'] = edge_matrix['kt'].T
            # kk:
            edge_matrix['kk'] = np.column_stack((edge_matrix['kk'], np.zeros((edge_matrix['kk'].shape[0], 1))))
            edge_matrix['kk'] = np.row_stack((edge_matrix['kk'], np.zeros((1, edge_matrix['kk'].shape[1]))))
                # 暂时只扩容，关系尚未添加


# 更改KIES矩阵内的值
def change_A_find_cluster(index, A, result, ptses):
    sim_list = []
    metapath_matrix = []
    for metapath in metapath_list:
        metapath_matrix.append(generate_Metapath_adjM(metapath))
    w = get_weight(weight_file)
    for j in range(A.shape[1]):
        A[index][j] = cal_KIES(w, index, j, metapath_matrix)
        A[j][index] = A[index][j]
        sim_list.append(A[index][j])
    # 判断index所属的聚类
    neighbor_list = []
    for i in range(len(sim_list)):
        if sim_list[i] > eps:
            neighbor_list.append(i)
    # 判断新点属于核心点、边界点还是噪音点
    density = len(neighbor_list)
    max_c = -1
    # 如果是核心点或density超过2的边界点，其聚类为邻域内最多的那一类
    if density > MinPts or (1 < density <= MinPts and density > 2):
        c_list = []
        for neighbor in neighbor_list:
            if ptses[neighbor] != 0:
                c_list.append(result[neighbor])
        temp = 0
        for c in c_list:
            if c_list.count(c) > temp:
                max_c = c
                temp = c_list.count(c)
        result[index] = max_c

    # 如果是边界点且density为2，则把噪音点改为边界点，新增一类
    elif MinPts >= density > 1 and density == 2:
        if neighbor_list[0] == index:
            other_node = neighbor_list[0]
        else:
            other_node = neighbor_list[1]
        # 若另一个点不是噪音点，则聚类与另一个点一致
        if ptses[other_node] != 0:
            max_c = result[other_node]
            result[index] = max_c
        else:
            for c in result.values():
                if c > max_c:
                    max_c = c
            ptses[other_node] = 2
            max_c += 1
            result[index] = max_c
            result[other_node] = max_c
    # 如果是噪音点，则更改ptses
    else:
        ptses[index] = 0
    return max_c


def new_instance_cluster(user, text, u_d, e_d, k_d, topic_word_list, all_topic_list, ptses, result, A):
    global s_index
    # 获取当前文本的各个type
    user_id, entity_list, keyword_list, topic_id = \
        extract_text_type(user, text, u_d, e_d, k_d, topic_word_list, all_topic_list)
    # 更改各个矩阵，直到A
    delete_old_node(s_index, ptses, result)
    change_edge_matrix(s_index, user_id, entity_list, keyword_list, topic_id, e_d, k_d, topic_word_list)
    target_c = change_A_find_cluster(s_index, A, result, ptses)
    # 全局下标加1
    s_index += 1
    if s_index == node_size:
        s_index = 0

    return target_c



if __name__ == "__main__":
    init_HIN(ori_dir, orifile_name)
    generateAllEdgeMatrix()
    '''
    # 初始化总邻接矩阵
    w = get_weight("data/results_new_22.txt")
    adj_matrix = np.load("data/22_adj_data.npy")
    A = calculate_A(w, adj_matrix)
    '''
    A = np.load('data/A_init2.npy')
    # 静态聚类
    result, noise_point, ptses = dbscan(A, eps, MinPts)
    old_result = result
    # show_dbscan_result(result, noise_point, "data/instance2label.txt")

    # 获取type值和id的对应关系
    u_d, e_d, k_d, t_d = init_dict(HIN_dir, userfile_name, entityfile_name, keywordfile_name, topicfile_name)
    # 获取topic预测所需的语料库
    topic_word_list, all_topic_list = get_topic_word_list(os.path.join(HIN_dir, topicfile_name))

    # 模拟流式
    n_u_l = []
    n_t_l = []
    with open("data/oridata.txt", 'r', encoding='utf-8-sig') as f:
        for line in f:
            text = line.strip().split('\t')
            n_u_l.append(text[0])
            n_t_l.append(text[1])

    label_dict = {}
    labels_true = []
    labels_pred = []
    with open("data/instance2label.txt", 'r', encoding='utf-8-sig') as f:
        for line in f:
            id = line.strip().split('\t')
            event = int(id[0].replace('i', ''))
            label = int(id[1].replace('y', ''))
            label_dict[event] = label
            labels_true.append(label)

    cluster_id = 1000
    for i in range(len(n_t_l)):
        test_c = new_instance_cluster(n_u_l[i], n_t_l[i], u_d, e_d, k_d, topic_word_list, all_topic_list, ptses, result, A)
        if test_c == -1:
            labels_pred.append(cluster_id)
            cluster_id += 1
        else:
            labels_pred.append(test_c)
        print(str(i) + '\t' + str(labels_pred[i]))

    # print(labels_true)
    # print(labels_pred)
    with open('data/result/label_pred2.txt', 'w', encoding='utf-8-sig') as f:
        f.write(str(labels_pred))

    # 同质性：每个群集只包含单个类的成员
    homogeneity = metrics.homogeneity_score(labels_true, labels_pred)
    print("同质性：" + str(homogeneity))
    # 完整性：给定类的所有成员都分配给同一个群集
    completeness = metrics.completeness_score(labels_true, labels_pred)
    print("完整性：" + str(completeness))
    # 两者的调和平均V-measure
    vm = metrics.v_measure_score(labels_true, labels_pred)
    print("V_measure：" + str(vm))
    # FMI=TP/(sqrt((TP+FP)(TP+FN)))
    # FMI = metrics.fowlkes_mallows_score(labels_true, labels_pred)
    # print("FMI：" + str(FMI))
    ARI = metrics.adjusted_rand_score(labels_true, labels_pred)
    print("ARI：" + str(ARI))
    NMI = metrics.normalized_mutual_info_score(labels_true, labels_pred)
    print("NMI：" + str(NMI))





