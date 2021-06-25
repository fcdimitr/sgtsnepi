## test_digits.py
# Test 3D SG-t-SNE embedding 

from julia.api import Julia
import requests
from scipy.io import mmread
from scipy.sparse import csc_matrix
import sys
import os.path
from os import path
import tarfile
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# SG-tSNE-Π setup
# Setup julia
jl = Julia(compiled_modules=False)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)  # Add path

from sgtsnepi import sgtsnepi

if path.exists("optdigits_10NN.tar.gz"):
    pass
else:
    print("Downloading and extracting file")
    # Download the matrix and label file from Tim Davids' matrix collection
    url = "https://suitesparse-collection-website.herokuapp.com/MM/ML_Graph/optdigits_10NN.tar.gz"
    r = requests.get(url, allow_redirects=True)
    open('optdigits_10NN.tar.gz', 'wb').write(r.content)

    # Unzip to the current folder
    tar = tarfile.open('./optdigits_10NN.tar.gz', 'r:gz')
    tar.extractall()
    tar.close()

# Reading out the matrix and the label

A = mmread('./optdigits_10NN/optdigits_10NN.mtx')
A = csc_matrix(A)
L = mmread('./optdigits_10NN/optdigits_10NN_label.mtx')
L = csc_matrix(L).toarray()

# generate the embedding
Y = sgtsnepi(A, d = 3, λ = 10, max_iter=500)

# distinguishable color map (10 clusters)
cmap = [[0.3000, 0.3000, 1.0000],
        [0.3000, 1.0000, 0.3000],
        [1.0000, 0.3000, 0.3000],
        [1.0000, 0.8552, 0.3000],
        [0.3000, 0.3000, 0.4931],
        [1.0000, 0.3000, 0.8069],
        [0.3000, 0.5655, 0.3000],
        [0.5897, 0.3966, 0.3000],
        [0.3000, 1.0000, 0.9034],
        [0.3000, 0.7586, 1.0000]]
newcmap = ListedColormap(cmap)

# draw scatter plot (3D)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=L, marker='.', cmap=newcmap)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

## AUTHOR
# Tiancheng Liu <tcliu@cs.duke.edu>
# Dimitris Floros <fcdimitr@auth.gr>
# 
