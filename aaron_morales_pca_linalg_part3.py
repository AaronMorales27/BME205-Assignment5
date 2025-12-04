import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

# ---- Load dataset ----
X = np.load('dogs_X.npy', allow_pickle=True)       # shape (1355, 784)
clades = np.load('dogs_clades.npy', allow_pickle=True)  # shape (1355,)


# NMF does not find a global minimum as the hessian of W and H can not BOTH be positive semi definite(no two can be convex)
# This approximated local minimum is dependent on an initial condition, and gradient descent from there(Learning rates also)
# Similiar to PCA in that it compresses a matrix into n most significant components
# ---- Apply NMF ----
n_components = 5
# smart initialization based on SVD

nmf_model = NMF(n_components=n_components, init='nndsvda', random_state=42, max_iter=1000)
# Each row(sample) descrubes how much these n components contribute to the dog
W = nmf_model.fit_transform(X)   # shape (1355, 5)
# The 5 ancestry basis paterns mapped to the original 784 genetic signatures
H = nmf_model.components_        # shape (5, 784)

# ---- Normalize rows of W to sum to 1 ----
W_norm = W / W.sum(axis=1, keepdims=True) # normalizes n ancestry patterns to sum to one

# ---- Combine clades and normalized W ----
df = pd.DataFrame(W_norm, columns=[f'ancestry{i+1}' for i in range(n_components)])
df['clade'] = clades

# ---- Compute mean ancestry per clade ----
clade_summary = df.groupby('clade').mean()

# ---- Sort clades alphabetically ----
clade_summary = clade_summary.sort_index()

# ---- Compute ancestry prevalence equally across clades ----
# (each clade gets equal weight, not each individual)
ancestry_means_equal = clade_summary.mean(axis=0)  # mean across clades
ancestry_order = ancestry_means_equal.sort_values(ascending=False).index

# ---- Reorder ancestry columns by prevalence ----
clade_summary = clade_summary[ancestry_order]

# ---- Rename ancestry columns according to prevalence order ----
clade_summary.columns = [f'ancestry{i+1}' for i in range(len(ancestry_order))]

# ---- Round to 2 decimals ----
clade_summary = clade_summary.round(2)

# ---- Save to TSV ----
clade_summary.to_csv('dogs_ancestry_summary.tsv', sep='\t', index=True)