import numpy as np



def compute_correlation_matrix(file_path):
    data = np.genfromtxt(file_path, delimiter=';')
    X = data[:, 1:] if data.shape[1] > 1 else data
    means = np.mean(X, axis=0)
    deviations = X - means
    cov_matrix = np.dot(deviations.T, deviations) / (X.shape[0] - 1)
    std_devs = np.sqrt(np.diag(cov_matrix))
    corr_matrix = cov_matrix / np.outer(std_devs, std_devs)
    return corr_matrix


def build_correlation_graph(corr_matrix, threshold):
    n = corr_matrix.shape[0]
    graph = {i: [] for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr_matrix[i, j]) >= threshold:
                graph[i].append((j, corr_matrix[i, j]))
                graph[j].append((i, corr_matrix[i, j]))

    return graph


def find_connected_components(graph):
    visited = set()
    components = []

    for node in graph:
        if node not in visited:
            stack = [node]
            component = []
            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    component.append(current)
                    for neighbor, _ in graph[current]:
                        if neighbor not in visited:
                            stack.append(neighbor)
            components.append(component)

    return components


def group_features(corr_matrix, threshold):
    graph = build_correlation_graph(corr_matrix, threshold)
    components = find_connected_components(graph)
    return components

file_path = "testA.csv"
try:
    np.set_printoptions(
        precision=3,
        suppress=True,
        linewidth=100
    )
    corr_matrix = compute_correlation_matrix(file_path)
    print("Корреляционная матрица:")
    print(corr_matrix)

    threshold = 0.63


    feature_groups = group_features(corr_matrix, threshold)
    print("\nГруппы признаков:")
    for i, group in enumerate(feature_groups, 1):
        print(f"Группа {i}: {[x + 1 for x in group]}")

except Exception as e:
    print(f"Произошла ошибка: {e}")
