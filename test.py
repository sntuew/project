import numpy as np

file_path = "test.csv"
data = np.genfromtxt(file_path,delimiter=";").astype(int)
avg = np.mean(data, axis = 0)
dev = data - avg
n = data.shape[0]
cov_matrix = np.zeros((data.shape[1], data.shape[1]))
for i in range(data.shape[1]):
    for j in range(data.shape[1]):
        cov = np.sum(dev[:, i] * dev[:, j]) / (n - 1)
        cov_matrix[i, j] = cov
std_dev = np.sqrt(np.diag(cov_matrix))
corr_matrix = np.zeros((data.shape[1], data.shape[1]))
for i in range(data.shape[1]):
    for j in range(data.shape[1]):
        if std_dev[i] != 0 and std_dev[j] != 0:
            corr = cov_matrix[i, j] / (std_dev[i] * std_dev[j])
        else:
            corr = 0
        corr_matrix[i, j] = corr
corr_matrix = np.round(corr_matrix,3)

R = 0.63
modified_corr_matrix = corr_matrix.copy()
np.fill_diagonal(modified_corr_matrix, -1)
modified_corr_matrix[modified_corr_matrix == 1] = 0.99
np.fill_diagonal(modified_corr_matrix, 1)
corr_matrix = modified_corr_matrix
def build_correlation_graph(corr_matrix):
    N = corr_matrix.shape[0]
    graph = {}
    remaining_indices = list(range(N))
    max_val = -1
    max_i, max_j = -1, -1
    for i in range(N):
        for j in range(N):
            if i != j and corr_matrix[i, j] > max_val:
                max_val = corr_matrix[i, j]
                max_i, max_j = i, j
    graph[max_i] = [(max_j, max_val)]
    graph[max_j] = [(max_i, max_val)]
    remaining_indices.remove(max_i)
    remaining_indices.remove(max_j)
    last_nodes = [max_i, max_j]
    while len(graph) < N:
        max_val = -1
        max_k = -1
        parent_node = -1
        for node in last_nodes:
            for k in remaining_indices:
                if corr_matrix[node, k] > max_val:
                    max_val = corr_matrix[node, k]
                    max_k = k
                    parent_node = node
        if max_k == -1:
            break
        if parent_node in graph:
            graph[parent_node].append((max_k, max_val))
        else:
            graph[parent_node] = [(max_k, max_val)]
        if max_k in graph:
            graph[max_k].append((parent_node, max_val))
        else:
            graph[max_k] = [(parent_node, max_val)]
        remaining_indices.remove(max_k)
        last_nodes = [parent_node, max_k]
    return graph
def split_into_groups(graph, threshold):
    visited = set()
    groups = []
    for node in graph:
        if node not in visited:
            group = []
            stack = [node]
            while stack:
                current_node = stack.pop()
                if current_node not in visited:
                    visited.add(current_node)
                    group.append(current_node)
                    for neighbor, weight in graph.get(current_node, []):
                        if weight >= threshold and neighbor not in visited:
                            stack.append(neighbor)
            if group:
                groups.append(group)
    all_nodes = set(graph.keys())
    for node in all_nodes - visited:
        groups.append([node])
    return groups
correlation_graph = build_correlation_graph(corr_matrix)
feature_groups = split_into_groups(correlation_graph, R)
print("Корреляционная матрица:")
print(corr_matrix)
print("\nГруппы признаков с порогом R =", R)
for i, group in enumerate(feature_groups, 1):
    print(f"Группа {i}: {group}")