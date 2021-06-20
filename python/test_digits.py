## test_digits.py
# Test 3D SG-t-SNE embedding 

## download the matrix and label file from Tim Davids' matrix collection
import requests
url = "https://suitesparse-collection-website.herokuapp.com/MM/ML_Graph/optdigits_10NN.tar.gz"
r = requests.get(url, allow_redirects=True)
open('optdigits_10NN.tar.gz', 'wb').write(r.content)


## unzip to the current folder
import tarfile

tar = tarfile.open('./optdigits_10NN.tar.gz', 'r:gz')
tar.extractall()
tar.close()

## reading out the matrix and the label
from scipy.io import mmread
from scipy.sparse import csc_matrix

A = mmread('./optdigits_10NN/optdigits_10NN.mtx')
A = csc_matrix(A)
L = mmread('./optdigits_10NN/optdigits_10NN_label.mtx')
L = csc_matrix(L).toarray()

## SG-tSNE-Π setup
# setup julia
from julia.api import Julia
jl = Julia(compiled_modules=False)

# add current path
import sys
sys.path.insert(0,"./")

# generate the embedding
from sgtsnepi import sgtsnepi
Y = sgtsnepi(A, d = 3, λ = 10, max_iter = 500)

# visualization
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

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
[0.3000, 0.7586, 1.0000]];
newcmap = ListedColormap(cmap)

# draw scatter plot (3D)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(Y[:,0], Y[:,1], Y[:,2], c=L, marker='.', cmap=newcmap)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

## AUTHOR
# Tiancheng Liu <tcliu@cs.duke.edu>
# Dimitris Foloros <fcdimitr@auth.gr>
# 
