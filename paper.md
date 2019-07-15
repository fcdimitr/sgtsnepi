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
large, sparse, stochastic graph into a $d$-dimensional space
($d = 1,2,3$) on a shared-memory computer. Sparse graphs or network data
emerge in numerous real-world applications and research fields, such as
biological networks, commercial product networks, food webs,
telecommunication networks, and word co-occurrence networks;
see [@Kovacs2019; @Yang2015] and references therein. Fundamental to many
graph analysis tasks, graph embedding maps the vertices of a graph to a
set of code/feature vectors in a code space. Often, a high-dimensional
graph embedding is accompanied by a $d$-dimensional embedding
($d = 1,2,3$) for visual inspection, interpretation of analysis results,
interactive exploration, and hypothesis generation.

SG-t-SNE-$\Pi$ is based on the algorithm SG-t-SNE [@Pitsianis2019] and
was used to obtain the published results of the latter.  The algorithm
is built upon precursors for $k$-nearest neighbor graph embedding, in
particular Stochastic Neighbor Embedding (SNE) [@Hinton2003],
t-Distributed Stochastic Neighbor Embedding (t-SNE) [@Maaten2008], and
their variants [@VanDerMaaten2014; @Linderman2019]. The precursor
algorithms typically require the data to be provided in a metric space
and the graph be regular with constant degree $k$. We remove this
limitation and enable low-dimensional embedding of arbitrary large,
sparse, stochastic graphs, which arise with real-world social and
commercial networks. The use of the precursor methods was practically
limited up to two-dimensional (2D) embeddings. We advocate 3D
embedding and make it fast and practical on modern laptop and desktop
computers, which are affordable and available to researchers and
developers by and large.

The SG-t-SNE-$\Pi$ library is implemented in `C++`. In addition, we
provide two particular types of interfaces. The first is to support the
conventional t-SNE, with its typical interface and
wrappers [@VanDerMaaten2014], which converts data points in a metric
space to a stochastic $k$NN graph. The second is a MATLAB interface for
SG-t-SNE-$\Pi$. The SG-t-SNE-$\Pi$ source code is archived with Zenodo [@Zenodo].


# References
