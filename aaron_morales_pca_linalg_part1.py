import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

'''

The eigenvectors rotate your coordinate system to align with directions of variance.

The eigenvalues tell you how much variance exists along each new axis.

PCA essentially performs that rotation and scaling automatically.

'''

X = np.load('part1_data.npy', allow_pickle=True) # (10 samples, 2 features) matrix
mu = X.mean(axis=0) # shape(features, ), 1D matrix storing average feature values
X_c = X - mu # shape(samples, features), mean of each column is zero i.e centerd
X = X.astype(np.float64)
n_samples, n_features = X.shape # 

# print(X)
# print(X_c)


pca = PCA(n_components=2)
pca.fit(X)

print("Sklearn direction of each PC (PC1, PC2):\n", pca.components_) # pca comps are eigenvectors,
print("Sklearn explained variance for each PC (PC1, PC2):\n", pca.explained_variance_) # respective eigen vectors
print("Sklearn fraction of variance explained by each PC (PC1, PC2):\n", pca.explained_variance_ratio_) # 

x_pca = pca.transform(X)
print("Sklearn data projection into 1-D (onto PC1):\n", x_pca[:, 0],'\n')

# [cov(x, x), cov(x, y)] cov(x, y) = cov(y, x)
# [cov(y, x), cov(y, y)] cov(y, y) = var(y) & cov(x, x) = var(x)
sigma = (X_c.T @ X_c) / (n_samples - 1) # shape (n_features, n_features) (2, 2)
print(f"Covariance Matrix\n{sigma}\n")

eigenvals, eigenvecs = np.linalg.eigh(sigma)
# Note: each column of eigenvecs is an eigenvector
# print(f"eigen values: \n{eigenvals},\neigen vectors: \n{eigenvecs}")

sorted_indices = np.argsort(eigenvals)[::-1] # slice reverses the indices for descending order

sorted_eigvals = eigenvals[sorted_indices]
sorted_eigvecs = eigenvecs[:, sorted_indices]

print(f"Eigen values: \n{sorted_eigvals},\nEigen vectors: \n{sorted_eigvecs}")
# See that V1 and l_1 is identical to sklearn output and V2 is antiparallel l_2 is identical
# The direction of V1 congruent to the plots direction
print("Question 1: See that both eigenvectors and eigenvalues are identical to the sklearn output,\nbut V2 is simply antiparallel.")
print("Question 2: According to the original plot and the dimensions of V1, this eigen vector is in\nthe direction of greatest variance.\n")
l_1 = sorted_eigvals[0]
l_2 = sorted_eigvals[1]

v_1 = sorted_eigvecs[:, 0]
v_2 = sorted_eigvecs[:, 1]
# print(f"eigen_vectors'{v_1}, {v_2}")

l_1_ratio = l_1 / sum(eigenvals)
l_2_ratio = l_2 / sum(eigenvals)

# l_ratio = l_1_ratio / l_2_ratio
print(f'Lambda 1 ratio {l_1_ratio}')
print(f'Lambda 2 ratio {l_2_ratio}')
# The ratios are identical to those of sklearn
print("Question 3: The ratios between sklearn and my own lambda ratios are identical and is expected\n")

projection = X_c @ v_1 # X_c * v_1, how far along our pr. comp. does each sample vector sit(squash to 1D)
# if we projected onto v_2, and plotted, this would be our PCA plot

print(f"Projection in the direction of V1\n{projection}")
print("projection_1D.png saved to files\nQuestion 4: The projection of our points in the direction of the largest PC vector is identical to sklearn.\n")
plt.figure(figsize=(8, 2))
plt.scatter(projection, np.zeros_like(projection), alpha=0.6)
plt.title("Projection onto First Principal Component (1D)")
plt.xlabel("Position along PC1")
plt.yticks([])  # hide y-axis (since it’s 1D)
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("projection_1D.png", dpi=150)
# plt.show()

print("Question 5: According to our original plot the distribution of our points" \
" along our first principal component is\ncongruent distribution of points along our projection numberline.")
