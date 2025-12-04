import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load our MNIST dataset(784 pixel feature columns w/ 6000 labeled numbers)
X = np.load('MNIST_X_subset.npy', allow_pickle=True)  # shape (6000, 784)
y = np.load('MNIST_y_subset.npy', allow_pickle=True)  # shape (6000,)
X = X / 255.0
'''
# Ensure float for means, preserve original range (MNIST is usually 0-255)
X = X.astype(np.float64)
n_samples, n_features = X.shape

# scale pixel values


'''

# Calculate mean for each feature vector(col wise) and center our data
mu = np.mean(X, axis=0)
X_c = X - mu

'''
# Introducing SVT
# PCA and SVD are mathematically equivalent.
# Any m x n matrix can be turned into two square and symetric
# S_L = AA^t of shape (m x m)
#   w/ m perpendicular eigen vectors in R^M - These are the 'left singular vectors of A'
# S_R = A^tA of shape (n x n)
#   w/ n perpendicular eigen vectors in R^N - These are the 'right singular vectors of A'
# S_L and S_R are ositive semi-definite matrices wherein they contain positive and
#    degenerate eigenvalues(share as many eigen values as dimensions, and non overlaps are zero)
# The sqrt(SR and LR common eigen values) are the singular values of matrix A

# What are the constituent components
# A = U S Vt
# Vt: Orthogonal (m x m) matrix contains the right singular vectors of matrix A(desc order) transposed
#   Orthogonal matrix applies a rotation such that the RSVs orientate to standard basis(i,j,k)
# Sigma: rectangular diagonal matrix(size m x n) containing singular values of A and zeros
#   A square diagonal matrix w/dim, erases degenerate dimension and stretchs by singular values
# U: Orthogonal (n x n) matrix contains the left singular vectors of matrix A(desc order)
#   Orthogonal matrix applies another rotation to re-align with the LSVs
'''

U, S, Vt = np.linalg.svd(X_c, full_matrices=False) # false is  asking for the compact SVD, which trims away all the redundant zero parts.
# U.shape = (6000, 784), would be 6k x 6k, but rank is atmost 784 dimensions
# S.shape = (784,) 1D array of singular values 
# Vt.shape = (784, 784), this is the full array of RSVs

# Project data onto first 2 PCs (2 largest principal componeonts/RSVs)
# takes the first two eigen vectors (2vectors x 784 features) transpose to (784 x 2)
# X_c (6000 x 784) * (784 x 2) Vt[:2].T, shape \/ \/ \/
X_pca = X_c @ Vt[:2].T  # shape (6000, 2)
# each of the 6k images in terms of its coordinates along PC1 and PC2.

# Plot 2D projection
plt.figure(figsize=(8,6))
sc = plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='tab10', s=10, alpha=0.7)
plt.colorbar(sc, label='Digit Label')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("MNIST PCA 2D (via SVD)")


plt.grid(True, linestyle='--', alpha=0.6)
plt.locator_params(axis='x', nbins=20)
plt.locator_params(axis='y', nbins=20)

plt.tight_layout()
plt.savefig("MNIST_PCA_2D.png", dpi=300)
plt.close()

# PART 4
x0_c = X_c[0]  # first centered image in our dataset
# Vt at 0 and 1 rows are the first two PComps/RSVs/eigenvectors
# X_pca[0], how far along our 1st 2 PComps this first image is, mutliplied by their vectors/direction of variance again
x0_reconstructed = (X_pca[0,0] * Vt[0]) + (X_pca[0,1] * Vt[1]) + mu # returned to original position by adding mean back
# same as pca.inverse_transform(X_pca[0].reshape(1, -1))

# PCA reconstruction only regards our 2 largest variance directions
# This lack of regard for fine detail low variance directions in feature space contributes to fuzzy reconstructions
# This subset of few PComps ->  smoother and fuzzier  reconstructions by reverse transform
# Reconstruct image
plt.imshow(X[0].reshape(28,28), cmap='gray') # X was already scaled 0-1(/255), so no vmax range param
plt.axis('off')
plt.title("Original Image")
plt.savefig("MNIST_original.png", dpi=300)
plt.close()

plt.imshow(x0_reconstructed.reshape(28,28), cmap='gray')
plt.axis('off')
plt.title("Reconstructed (2 PCs, manual SVD)")
plt.savefig("MNIST_reconstructed_2PC.png", dpi=300)
plt.close()

# PART 5

# Pick a coordinate from the MNIST_PCA_2D.png
chosen_coord = np.array([-3.3, 2.5]) # 1
# chosen_coord = np.array([6, 0.5]) # 0
# chosen_coord = np.array([1.5, 3.5]) # 3
# Reconstruct this 2D coordinate back into the original 784-D image space
x_reconstructed_from_coord = chosen_coord[0] * Vt[0] + chosen_coord[1] * Vt[1] + mu
# Visualize and save the reconstructed image
plt.imshow(x_reconstructed_from_coord.reshape(28, 28), cmap='gray')
plt.axis('off')
plt.title("Reconstructed Digit (from chosen 2D coordinate)")
plt.savefig("MNIST_reconstructed_1_from_coord.png", dpi=300)
plt.close()