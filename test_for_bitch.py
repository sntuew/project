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