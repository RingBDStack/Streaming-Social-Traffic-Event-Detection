# 所需输入：权重文件results_new.txt; 邻接矩阵文件adj_data.npy; label文件instance2label.txt

import numpy as np
from sklearn import metrics

def get_weight(weight_file = "data/results_new_22.txt"):
    w_list = list()
    with open(weight_file, 'r', encoding='utf-8-sig') as f:
        for line in f:
            if '[' in line:
                w_str = line.replace('[', '').replace(']', '').replace("weight:", '').strip()
                w_float = float(w_str)
                w_list.append([w_float])
    w = np.array(w_list)
    return w


def calculate_A(w, adj_matrix):
    adj_size = adj_matrix.shape
    meta_size = adj_size[0]
    event_size = adj_size[1]
    A = np.zeros((event_size, event_size))
    for i in range(event_size):
        for j in range(event_size):
            sum = 0
            for k in range(meta_size):
                sum += w[k][0] * adj_matrix[k][i][j]
            A[i][j] = sum
    return A


def dbscan(A, eps, MinPts):
    ptses = []
    for i in range(A.shape[0]):
        density = 0
        for j in range(A.shape[1]):
            if A[i][j] > eps:
                density += 1
        # print(str(i) + '\t' + str(density))
        if density > MinPts:
            # 核心点（Core Points）
            # 空间中某一点的密度，如果大于某一给定阈值MinPts，则称该为核心点
            pts = 1
        elif MinPts >= density > 1:
            # 边界点（Border Points）
            # 空间中某一点的密度，如果小于某一给定阈值MinPts，则称该为边界点
            pts = 2
        else:
            # 噪声点（Noise Points）
            # 数据集中不属于核心点，也不属于边界点的点，也就是密度值为1的点
            pts = 0
        ptses.append(pts)

    # 把噪声点过滤掉，因为噪声点无法聚类，它们独自一类
    corePoints = list()
    noise_point = []
    for i in range(len(ptses)):
        if ptses[i] != 0:
            corePoints.append(i)
        else:
            noise_point.append(i)

    # 首先，把每个点的邻域都作为一类
    # 邻域（Neighborhood）
    # 空间中任意一点的邻域是以该点为圆心、以 Eps 为半径的圆区域内包含的点集合
    cluster = dict()
    i = 0
    for event in corePoints:
        near_list = list()
        for j in range(A.shape[1]):
            if A[event][j] > eps:
                near_list.append(j)
        cluster[i] = np.array(near_list)
        i += 1

    # 然后，将有交集的邻域，都合并为新的邻域
    for i in range(len(cluster)):
        for j in range(len(cluster)):
            if len(set(cluster[j]) & set(cluster[i])) > 0 and i != j:
                cluster[i] = list(set(cluster[i]) | set(cluster[j]))
                cluster[j] = list()

    # 最后，找出独立（也就是没有交集）的领域，就是最后的聚类的结果了
    result = dict()
    j = 0
    for i in range(len(cluster)):
        if len(cluster[i]) > 0:
            result[j] = cluster[i]
            j = j + 1

    # 从<类号，[事件0，事件1，……]> 改为 <事件0，类号>，<事件1，类号>
    result_reverse = {}
    for key in result.keys():
        for id in result[key]:
            result_reverse[id] = key

    return result_reverse, noise_point, ptses


def show_KIES(A, label_dict):
    event_list = []
    label_list = []
    for keys in label_dict.keys():
        event_list.append(keys)
        label_list.append(label_dict[keys])
    for i in range(len(label_list)):
        for j in range(i+1,len(label_list)):
            if label_list[i] == label_list[j]:
                event1 = int(event_list[i])
                event2 = int(event_list[j])
                print(str(event1) + '\t' + str(event2) + '\t' + str(A[event1, event2]))


def show_dbscan_result(result, noise_point, i2l_path="data/instance2label.txt"):
    label_dict = {}
    with open(i2l_path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            id = line.strip().split('\t')
            event = int(id[0].replace('i', ''))
            label = int(id[1].replace('y', ''))
            label_dict[event] = label

    cluster = set()
    for i in result.keys():
        cluster.add(result[i])
    cluster_list = list(cluster)
    cluster_dict = {}
    for cluster_id in cluster_list:
        cluster_dict[cluster_id] = list()
    for i in result.keys():
        cluster_dict[result[i]].append(i)
    for key in cluster_dict.keys():
        event_list = []
        for event in cluster_dict[key]:
            event_list.append(label_dict[event])
        print(str(key) + '\t' + str(len(event_list)) + '\t' + str(event_list))
    print("class num: " + str(len(cluster_list)))
    print("noise num: " + str(len(noise_point)))

    labels_true = []
    labels_pred = []

    for key in label_dict:
        labels_true.append(label_dict[key])
        try:
            labels_pred.append(result[key])
        except:
            labels_pred.append(int(-1))

    # 同质性：每个群集只包含单个类的成员
    homogeneity = metrics.homogeneity_score(labels_true, labels_pred)
    print("同质性：" + str(homogeneity))
    # 完整性：给定类的所有成员都分配给同一个群集
    completeness = metrics.completeness_score(labels_true, labels_pred)
    print("完整性：" + str(completeness))
    # 两者的调和平均V-measure
    vm = metrics.v_measure_score(labels_true, labels_pred)
    print("V_measure：" + str(vm))
    ARI = metrics.adjusted_rand_score(labels_true, labels_pred)
    print("ARI：" + str(ARI))
    NMI = metrics.normalized_mutual_info_score(labels_true, labels_pred)
    print("NMI：" + str(NMI))


if __name__ == "__main__":
    '''
    w = get_weight()
    adj_matrix = np.load("data/22_adj_data.npy")
    A = calculate_A(w, adj_matrix)
    np.save("data/A.npy", A)
    print(A)
    print(A.shape)
    '''
    A = np.load("data/A_init.npy")
    eps = 8.7
    MinPts = 2
    # result:字典，key=聚类id，value=事件id的list；
    # noise_point: 噪音事件列表
    # ptses: 事件是否为噪音点的标记列表，0噪音，1核心，2边界
    result, noise_point, ptses = dbscan(A, eps, MinPts)
    # print(result)

    # print(ptses)
    event_map = {}
    with open("data/init_map.txt", 'r', encoding='utf-8-sig') as f:
        for line in f:
            id = line.strip().split('\t')
            a_event = int(id[0])
            i_event = int(id[1].replace('i', ''))
            event_map[i_event] = a_event

    label_dict = {}
    with open("data/instance2label_init.txt", 'r', encoding='utf-8-sig') as f:
        for line in f:
            id = line.strip().split('\t')
            event = int(id[0].replace('i', ''))
            label = int(id[1].replace('y', ''))
            label_dict[event_map[event]] = label

    cluster = set()
    for i in result.keys():
        cluster.add(result[i])
    cluster_list = list(cluster)
    cluster_dict = {}
    for cluster_id in cluster_list:
        cluster_dict[cluster_id] = list()
    for i in result.keys():
        cluster_dict[result[i]].append(i)
    for key in cluster_dict.keys():
        event_list = []
        for event in cluster_dict[key]:
            event_list.append(label_dict[event])
        print(str(key) + '\t' + str(len(event_list))+ '\t' +str(event_list))
    print("class num: " + str(len(cluster_list)))
    print("noise num: " + str(len(noise_point)))

    labels_true = []
    labels_pred = []

    cluster_id = 1000
    for key in label_dict:
        labels_true.append(label_dict[key])
        try:
            labels_pred.append(result[key])
        except:
            labels_pred.append(int(cluster_id))
            cluster_id += 1

    print(labels_true)
    print(labels_pred)

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

    # show_KIES(A, label_dict)


