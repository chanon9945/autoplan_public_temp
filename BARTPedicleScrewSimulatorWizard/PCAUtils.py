import numpy as np
import scipy as sp
from sklearn.decomposition import PCA

def apply_pca(volume):
    pca = PCA(n_components=3)
    pca.fit(volume)
    coeff = pca.components_
    latent = pca.explained_variance_ratio_
    score = pca.transform(volume)
    return coeff,latent,score

    