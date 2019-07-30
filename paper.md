---
title: 'SG-t-SNE-Π: Swift Neighbor Embedding of Sparse Stochastic Graphs'
tags:
  - graph embedding
  - stochastic neighbor embedding
  - t-SNE
  - data analysis
  - machine learning
  - C++
authors:
  - name: Nikos Pitsianis
    orcid: 0000-0002-7353-3524
    affiliation: "1,2"
  - name: Dimitris Floros
    orcid: 0000-0003-1190-4075
    affiliation: 1
  - name: Alexandros-Stavros Iliopoulos
    orcid: 0000-0002-1959-9792
    affiliation: 2
  - name: Xiaobai Sun
    affiliation: 2
affiliations:
  - name: Department of Electrical and Computer Engineering, Aristotle University of Thessaloniki, Thessaloniki 54124, Greece
    index: 1
  - name: Department of Computer Science, Duke University, Durham, NC 27708, USA
    index: 2
date: 14 July 2019
bibliography: references.bib
---


# Summary

SG-t-SNE-$\Pi$ is a high-performance software for swift embedding of a
large, sparse, stochastic graph/network into a $d$-dimensional space
($d = 1,2,3$) on a shared-memory computer, especially on personal laptop
and desktop computers. Graphs/networks are an important type of
relational data, arising ubiquitously in real-world applications and
various research fields. Such data include biological networks, social networks,
communication networks, food webs, word co-occurrence networks, 
see @Kovacs2019 and @Yang2015 for more real-world networks. Graph embedding
maps each vertex of the graph to a $d$-dimensional feature vector. Graph
embedding into a $d$-dimensional space with $d=1,2,3$ is frequently used
in data-based scientific studies for visual inspection of data,
interpretation of network-based analysis results, interactive inquiries
and hypothesis generation.

The software SG-t-SNE-$\Pi$ and its underlying algorithm are built upon
precursor algorithms and software for stochastic neighbor embedding of
high-dimensional data, namely the original Stochastic Neighbor
Embedding (SNE) algorithm by @Hinton2003,
the algorithm for t-distributed Stochastic Neighbor Embedding (t-SNE) by
@Maaten2008, and their
variants [@VanDerMaaten2014; @Linderman2019].[^1][^2] The t-SNE
algorithm has successfully assisted scientific discoveries, as reported
in numerous articles in Nature and Science magazines. However, previous
t-SNE algorithms and software are limited in two aspects:
(i) The algorithms require that the data points be in a metric space and the
associated graph (internally generated) be regular with a constant
degree. In many real-world networks, the vertices do not readily reside
in a metric space, and their degrees vary greatly, far from
constant.
(ii) The software is limited in practical use either to small graphs/networks
or to embedding to $d<3$ dimensional space.
We remove both limitations. SG-t-SNE-$\Pi$ admits arbitrary,
sparse, stochastic graphs/networks. It is demonstrated by @Pitsianis2019 for
novel, autonomous embedding of large, real-world stochastic networks.
SG-t-SNE-$\Pi$ also enables fast three-dimensional (3D) graph embedding,
which preserves and reveals more or even critical structural information
as shown by @Pitsianis2019, on modern laptop and desktop computers
with ease of use.

SG-t-SNE-$\Pi$ is implemented in C++. It takes as input a stochastic
graph and outputs $d$-dimensional coordinate vectors. We provide two additional interfaces. The
first is to support the conventional t-SNE, with its typical interface
and wrappers [@VanDerMaaten2014], which converts data points in a metric
space to a stochastic $k$-nearest neighbor graph. The second is a MATLAB
interface. SG-t-SNE-$\Pi$ is used to obtain all numerical experiments in
the research article by @Pitsianis2019 and the accompanying supplementary
material.[^3]

[^1]: <https://github.com/lvdmaaten/bhtsne>

[^2]: <https://github.com/KlugerLab/FIt-SNE>

[^3]: <http://t-sne-pi.cs.duke.edu>

# References
