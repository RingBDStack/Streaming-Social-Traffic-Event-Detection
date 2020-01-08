# 计算相似度矩阵
import numpy as np
import os

type_list = ['i', 'e', 'k', 'u', 't']  # 五个type：事件实例、实体、关键词、用户、主题
path_matrix_str_list = ['uu', 'ee', 'kk', 'tt', 'ui', 'ie', 'ik', 'it', 'ke', 'kt']

metapath_list = ['iui', 'iei', 'iki', 'iti', 'iuui', 'ieei', 'ikki', 'itti', 'iekei', 'ikeki', 'iktki', 'itkti',
                 'iekkei', 'ikeeki', 'ikttki', 'itkkti', 'ieekeei', 'ikkekki', 'ikktkki', 'ittktti', 'iektkei',
                 'itkekti'] 


den_w = len(metapath_list)
input_data_dir = "data/HINdata"    # 数据文件根目录
output_data_dir = "data/GCNData"

type_num_dict = {}      # 各个type的数量
edge_matrix = {}        # 存放各超边对应的邻接矩阵
metapath_matrix = {}    # 存放元路径矩阵


def getTypeNum(sub_path):  # 获取某个type的个数
    with open(os.path.join(input_data_dir, sub_path), 'r', encoding='utf-8-sig') as f:
        count = 0
        for line in f:
            count += 1
    return count


def getAllTypeNum():  # 获取全部type的个数，存入字典
    type_num_dict['i'] = getTypeNum("instance2id.txt")
    type_num_dict['e'] = getTypeNum("entity2id.txt")
    type_num_dict['k'] = getTypeNum("keyword2id.txt")
    type_num_dict['u'] = getTypeNum("user2id.txt")
    type_num_dict['t'] = getTypeNum("topic2id.txt")


def generate_edge_matrix_from_file(sub_path, type1, type2):  # 生成两个type之间的邻接矩阵（01，不对称）
    matrix_w = np.zeros((type_num_dict[type1], type_num_dict[type2]))
    mat_index_list = []
    with open(os.path.join(input_data_dir, sub_path), 'r', encoding='utf-8-sig') as f:
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
    mul = np.eye(size, size)
    for edge in edge_list:
        mul = np.matmul(mul, edge_matrix[edge])
    return mul
        # np.save(os.path.join(data_dir,'sim/sim_' + str(metapath) + '.npy'), mul)


def know_sim(M, i, j):
    broadness = M[i][i] + M[j][j]
    overlap = 2*M[i][j]
    if broadness == 0:
        return 0
    else:
        return overlap/broadness


def generate_KIESMatrix_list():
    for k in range(len(metapath_list)):
        path = metapath_list[k]
        M = generate_Metapath_adjM(str(path))
        event_num = type_num_dict['i']
        sim = np.zeros((event_num, event_num))
        print('path' + str(k) + ' M calculation finished.')
        for i in range(event_num):
            sim[i][i] = 1
        for i in range(event_num - 1):
            for j in range(i + 1, event_num):
                sim[i][j] = know_sim(M, i, j)
                sim[j][i] = sim[i][j]
        del M
        np.save(os.path.join(output_data_dir,'sim/sim_' + str(path) + '.npy'), sim)  # 存储的为单一元路径的相似度矩阵
        del sim
        print('path' + str(k) + ' sim matrix finished.')


# 主程序
if __name__ == "__main__":
    getAllTypeNum()
    generateAllEdgeMatrix()
    # print(generate_Metapath_adjM('itkekti'))
    generate_KIESMatrix_list()

    l = []
    n = 0
    for i in range(len(metapath_list)):
        f = np.load(os.path.join(output_data_dir, 'sim/sim_' + metapath_list[i] + '.npy'))

        # 源代码中这里有一段乘对角化矩阵的注释，详见adj_matrix.py

        l.append(f)
    adj_data = np.array(l)

    # print(adj_data[4])
    print(adj_data.shape)
    np.save(os.path.join(output_data_dir,"22_adj_data.npy"), adj_data)  # 最后是一个数组array，数组内元素为矩阵

