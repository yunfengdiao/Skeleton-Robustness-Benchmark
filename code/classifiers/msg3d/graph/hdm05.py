import sys
sys.path.insert(0, '')
sys.path.extend(['../'])
from classifiers.msg3d.graph import tools

num_node = 25
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(10, 11), (11, 12), (12, 13), (13, 14),
                              (13, 15), (15, 16), (16, 17), (17, 18), (18, 19),
                              (13, 20), (20, 21), (21, 22), (22, 23), (23, 24),
                              (10, 0), (0, 1), (1, 2), (2, 3), (3, 4),
                              (10, 5), (5, 6), (6, 7), (7, 8), (8, 9)]
inward = [(i , j ) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class AdjMatrixGraph:
    def __init__(self, *args, **kwargs):
        self.edges = neighbor
        self.num_nodes = num_node
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A_binary = tools.get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = tools.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)
        self.A = tools.normalize_adjacency_matrix(self.A_binary)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    graph = AdjMatrixGraph()
    A, A_binary, A_binary_with_I = graph.A, graph.A_binary, graph.A_binary_with_I
    f, ax = plt.subplots(1, 3)
    ax[0].imshow(A_binary_with_I, cmap='gray')
    ax[1].imshow(A_binary, cmap='gray')
    ax[2].imshow(A, cmap='gray')
    plt.show()
    print(A_binary_with_I.shape, A_binary.shape, A.shape)