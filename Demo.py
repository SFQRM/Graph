import networkx as nx
import numpy as np


def readNetwork(filename, N):
    fin = open(filename, 'r')                               # 读文件
    A_matrix = np.zeros((N, N), dtype=int)                  # 创建一个NxN的全零矩阵，数据类型为int型
    matrix_row = 0                                          # 定义矩阵的行，从第0行开始
    for line in fin.readlines():                            # 一次性读取所有行，并存储为字符串列表
        list = line.strip('\n').split(' ')                  # 处理逐行数据：strip表示把头尾的'\n'去掉，split表示以空格来分割行数据，
        # list = line.strip('\n').split('\t')
        A_matrix[matrix_row] = list[0:N]                    # list[0:8]表示列表的0~7列数据放到矩阵中的matrix_row行
        matrix_row += 1
    return A_matrix


def selfLoops(matrix):
    I_matrix = np.eye(matrix.shape[0])
    return np.add(matrix, I_matrix)


def relu(x):
    s = np.where(x < 0, 0, x)
    return s


def gcn_layer(A, D_hat, X, W):
    return relu(D_hat**-1 * A * X * W)


if __name__ == '__main__':
    # G = nx.Graph()
    dirPath = 'bridge node network.data'
    N = 8                                                   # N: 邻接矩阵维度

    A = readNetwork(dirPath, N)
    # A = np.matrix([                                         # A:邻接矩阵
    #     [0, 1, 0, 0],
    #     [0, 0, 1, 1],
    #     [0, 1, 0, 0],
    #     [1, 0, 1, 0]],
    #     dtype=float                                         # 定义类型
    # )
    # print("邻接矩阵A：\n",A)

    X = np.matrix([                                         # X:特征矩阵
        [i, -i]
        for i in range(1, A.shape[0]+1)
    ], dtype=float)                                         # 定义类型
    # print(A * X)

    I = np.eye(A.shape[0])
    # A_hat = selfLoops(A)                                    # A_hat: 对邻接矩阵加一个单位矩阵
    # print(A_hat)

    # D = np.array(np.sum(A, axis=0))[0]  # D:A的度序列
    # D = np.matrix(np.diag(D))  # D:A的（入）度矩阵

    # D_hat = np.array(np.sum(A_hat, axis=0))                 # D_hat:A_hat的度序列
    D_hat = np.array(np.sum(A, axis=0))                     # D_hat:A_hat的度序列
    D_hat = np.matrix(np.diag(D_hat))                       # D_hat:A的（入）度矩阵

    W = np.matrix([                                         # W: 权重矩阵
        [1],
        [-1]
    ])

    # print(D_hat**-1 * A * X * W)
    # print(relu(D_hat**-1 * A_hat * X * W))

    W_1 = np.random.normal(
        loc=0, scale=1, size=(N, 4)
    )
    W_2 = np.random.normal(
        loc=0, size=(W_1.shape[1], 2)
    )
    W_3 = np.random.normal(
        loc=0, size=(W_2.shape[1], 1)
    )

    H_1 = gcn_layer(A, D_hat, I, W_1)
    H_2 = gcn_layer(A, D_hat, H_1, W_2)
    H_3 = gcn_layer(A, D_hat, H_2, W_3)
    # H_1 = gcn_layer(A_hat, D_hat, I, W_1)
    # H_2 = gcn_layer(A_hat, D_hat, H_1, W_2)

    output = H_3
    print(output)

    # feature_representations = {
    #     node: np.array(output)[node]
    #     for node in range(N)
    # }
    # print(feature_representations)
