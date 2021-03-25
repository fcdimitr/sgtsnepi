# SG-t-SNE-Π <br/> Swift Neighbor Embedding of Sparse Stochastic Graphs

[![DOI](http://joss.theoj.org/papers/10.21105/joss.01577/status.svg)](https://doi.org/10.21105/joss.01577)
[![DOI](https://zenodo.org/badge/196885143.svg)](https://zenodo.org/badge/latestdoi/196885143)
[![GitHub license](https://img.shields.io/github/license/fcdimitr/sgtsnepi.svg)](https://github.com/fcdimitr/sgtsnepi/blob/master/LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/fcdimitr/sgtsnepi.svg)](https://github.com/fcdimitr/sgtsnepi/issues/)

-   [Overview](#overview)
    -   [Precursor algorithms](#precursor-algorithms)
    -   [Approximation of the gradient](#approximation-of-the-gradient)
    -   [SG-t-SNE-Π](#sg-t-sne-π)
        -   [Accelerated accumulation of attractive
            interactions](#accelerated-accumulation-of-attractive-interactions)
        -   [Accelerated accumulation of repulsive
            interactions](#accelerated-accumulation-of-repulsive-interactions)
        -   [Rapid intra-term and inter-term data
            relocations](#rapid-intra-term-and-inter-term-data-relocations)
    -   [Supplementary material](#supplementary-material)
-   [References](#references)
-   [Getting started](#getting-started)
    -   [System environment](#system-environment)
    -   [Prerequisites](#prerequisites)
    -   [Installation](#installation)
        -   [Basic instructions](#basic-instructions)
        -   [Support of the conventional t-SNE](#support-of-the-conventional-t-sne)
        -   [MATLAB interface](#matlab-interface)
    -   [Usage demo](#usage-demo)
-   [License and community guidelines](#license-and-community-guidelines)
-   [Contributors](#contributors)

## Overview 

We introduce SG-t-SNE-Π, a high-performance software for swift
embedding of a large, sparse, stochastic graph
<img src="svgs/4ad232c35b5cd188a13d128bb2c1eecc.svg" align=middle width=103.24177049999999pt height=24.65753399999998pt/> into a <img src="svgs/2103f85b8b1477f430fc407cad462224.svg" align=middle width=8.55596444999999pt height=22.831056599999986pt/>-dimensional space
(<img src="svgs/3fa1f779de09763d248814c0c4f40d07.svg" align=middle width=69.74298869999998pt height=22.831056599999986pt/>) on a shared-memory computer. The algorithm SG-t-SNE and the
software t-SNE-Π were first described in Reference [[1](#Pitsianis2019)].
The algorithm is built upon precursors for embedding a <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg" align=middle width=9.075367949999992pt height=22.831056599999986pt/>-nearest
neighbor (<img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg" align=middle width=9.075367949999992pt height=22.831056599999986pt/>NN) graph, which is distance-based and regular with
constant degree <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg" align=middle width=9.075367949999992pt height=22.831056599999986pt/>. In practice, the precursor algorithms are also
limited up to 2D embedding or suffer from overly long latency in 3D
embedding. SG-t-SNE removes the algorithmic restrictions and enables
<img src="svgs/2103f85b8b1477f430fc407cad462224.svg" align=middle width=8.55596444999999pt height=22.831056599999986pt/>-dimensional embedding of arbitrary stochastic graphs, including, but
not restricted to, <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg" align=middle width=9.075367949999992pt height=22.831056599999986pt/>NN graphs. SG-t-SNE-Π expedites the
computation with high-performance functions and materializes 3D
embedding in shorter time than 2D embedding with any precursor algorithm
on modern laptop/desktop computers.

### Precursor algorithms 

The original t-SNE [[2](#Maaten2008)] has given rise to several variants. Two
of the variants, [t-SNE-BH](https://lvdmaaten.github.io/tsne/) [[3](#VanDerMaaten2014)] and
[FIt-SNE](https://github.com/KlugerLab/FIt-SNE) [[4](#Linderman2019)], are distinctive and representative in their
approximation approaches to reducing algorithmic complexity. They are,
however, limited to <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg" align=middle width=9.075367949999992pt height=22.831056599999986pt/>NN graph embedding. Specifically, at the user
interface, a set of <img src="svgs/8da1486405f46444193f741ce5a271d6.svg" align=middle width=54.15898619999999pt height=24.65753399999998pt/> data points,
<img src="svgs/4848cd4cae94c484c6f7fa961a7dc7d8.svg" align=middle width=92.14043849999999pt height=24.65753399999998pt/>, is provided in terms of their
feature vectors <img src="svgs/796df3d6b2c0926fcde961fd14b100e7.svg" align=middle width=16.08162434999999pt height=14.611878600000017pt/> in an <img src="svgs/ddcb483302ed36a59286424aa5e0be17.svg" align=middle width=11.18724254999999pt height=22.465723500000017pt/>-dimensional vector space
equipped with a metric/distance function. The input parameters include
<img src="svgs/2103f85b8b1477f430fc407cad462224.svg" align=middle width=8.55596444999999pt height=22.831056599999986pt/> for the embedding dimension, <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg" align=middle width=9.075367949999992pt height=22.831056599999986pt/> for the number of near-neighbors,
and <img src="svgs/6dbb78540bd76da3f1625782d42d6d16.svg" align=middle width=9.41027339999999pt height=14.15524440000002pt/> for the perplexity. A t-SNE algorithm maps the data points
<img src="svgs/38e077ab3accd56221488b5b38e548f4.svg" align=middle width=14.132466149999988pt height=22.465723500000017pt/> to data points <img src="svgs/0e8e2b1ee7fa5154c9090eadff7d254d.svg" align=middle width=90.34592984999999pt height=24.65753399999998pt/> in a
<img src="svgs/2103f85b8b1477f430fc407cad462224.svg" align=middle width=8.55596444999999pt height=22.831056599999986pt/>-dimensional space.

There are two basic algorithmic stages in a conventional t-SNE
algorithm. In the preprocessing stage, the <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg" align=middle width=9.075367949999992pt height=22.831056599999986pt/>NN graph is generated from
the feature vectors <img src="svgs/1bf183da1f64359241e53d08083c380e.svg" align=middle width=33.341937749999985pt height=24.65753399999998pt/> according to the metric function
and input parameter <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg" align=middle width=9.075367949999992pt height=22.831056599999986pt/>. Each data point is associated with a graph
vertex. Next, the <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg" align=middle width=9.075367949999992pt height=22.831056599999986pt/>NN graph is cast into a stochastic one,
<img src="svgs/b9d2acdd4ca9fbd1fb5226466229e0c0.svg" align=middle width=74.39104034999998pt height=24.65753399999998pt/>, and symmetrized to
<img src="svgs/8e2cb982230037ec6084a327c96d64e7.svg" align=middle width=63.820236149999985pt height=24.65753399999998pt/>, 

<p align="center"><img src="svgs/a86f48e97b076ed0bb6dfeb700fea86d.svg" align=middle width=536.6954290499999pt height=43.13440065pt/></p>

 where
<img src="svgs/64814475919e9b203cce2479fd26855d.svg" align=middle width=65.60868104999999pt height=24.65753399999998pt/> is the binary-valued adjacency matrix of the
<img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg" align=middle width=9.075367949999992pt height=22.831056599999986pt/>NN graph, with zero diagonal elements (i.e., the graph has no
self-loops), and <img src="svgs/2ccee0b78bcd2047f49b3b53962d20a4.svg" align=middle width=19.311372299999988pt height=22.831056599999986pt/> is the distance between <img src="svgs/c416d0c6d8ab37f889334e2d1a9863c3.svg" align=middle width=14.628015599999989pt height=14.611878600000017pt/> and
<img src="svgs/796df3d6b2c0926fcde961fd14b100e7.svg" align=middle width=16.08162434999999pt height=14.611878600000017pt/>. The Gaussian parameters <img src="svgs/e61ae7f2cb94c8418c30517775fde77d.svg" align=middle width=14.04400634999999pt height=14.15524440000002pt/> are determined by the
point-wise equations related to the same perplexity value <img src="svgs/6dbb78540bd76da3f1625782d42d6d16.svg" align=middle width=9.41027339999999pt height=14.15524440000002pt/>,


<p align="center"><img src="svgs/210a56bcf471bf05338f2704cbee406c.svg" align=middle width=424.26428549999997pt height=38.89287435pt/></p>



The next stage is to determine and locate the embedding coordinates
<img src="svgs/9f88162984437603577c62bf6f319682.svg" align=middle width=98.79627284999998pt height=24.65753399999998pt/> by minimizing the
Kullback-Leibler divergence 

<p align="center"><img src="svgs/8a31e14108bb84cf778034dddfdf62bd.svg" align=middle width=359.40184665pt height=24.0502251pt/></p>


where matrix <img src="svgs/3c30cbd100ce5ab96f26034f73886774.svg" align=middle width=89.28989189999999pt height=24.65753399999998pt/> is made of the
ensemble <img src="svgs/69b7d411a46d4a8f33a3ed4f1937f0c5.svg" align=middle width=12.337954199999992pt height=22.465723500000017pt/> regulated by the Student t-distribution,


<p align="center"><img src="svgs/0715f18ccb81de1afebc36e276b851da.svg" align=middle width=534.2742883499999pt height=43.7234787pt/></p>


In other words, the objective of
(3) is to find the optimal
stochastic matching between <img src="svgs/384591906555413c452c93e493b2d4ec.svg" align=middle width=12.92230829999999pt height=22.55708729999998pt/> and <img src="svgs/61ccc6d099c3b104d8de703a10b20230.svg" align=middle width=14.20083224999999pt height=22.55708729999998pt/> defined,
respectively, over the feature vector set <img src="svgs/38e077ab3accd56221488b5b38e548f4.svg" align=middle width=14.132466149999988pt height=22.465723500000017pt/> and the embedding
coordinate set <img src="svgs/69b7d411a46d4a8f33a3ed4f1937f0c5.svg" align=middle width=12.337954199999992pt height=22.465723500000017pt/>. The optimal matching is obtained numerically
by applying the gradient descent method. A main difference among the
precursor algorithms lies in how the gradient of the objective function
is computed.

### Approximation of the gradient 

The computation per iteration step is dominated by the calculation of
the gradient. Van der Maaten reformulated the gradient into two
terms [[3](#VanDerMaaten2014)]:


<p align="center"><img src="svgs/0ee6b599fe43800a72b9c4450d8862a7.svg" align=middle width=521.24979885pt height=71.90601659999999pt/></p>


The attractive interaction term can be cast as the sum of <img src="svgs/7f926a99555bec4c6525305cdea81193.svg" align=middle width=16.77517379999999pt height=22.831056599999986pt/>
matrix-vector products with the sparse matrix
<img src="svgs/d28f302ed1f97b242a449f0316aa2773.svg" align=middle width=96.93665849999998pt height=24.65753399999998pt/>. The vectors are composed of the
embedding coordinates, one in each dimension. The repulsive interaction
term can be cast as the sum of <img src="svgs/7f926a99555bec4c6525305cdea81193.svg" align=middle width=16.77517379999999pt height=22.831056599999986pt/> matrix-vector products with the
dense matrix <img src="svgs/8dcd6cb8c986db1c1c73c2046dc7863b.svg" align=middle width=97.28293244999999pt height=24.65753399999998pt/>. For clarity, we simply
refer to the two terms as the <img src="svgs/10113745b95b6c955f7c8b1959d3e01d.svg" align=middle width=27.12314054999999pt height=22.55708729999998pt/> term and the <img src="svgs/c6e87bbfdef510c8c42806f373d05c8f.svg" align=middle width=28.401664499999992pt height=22.55708729999998pt/>
term, respectively.

The <img src="svgs/c6e87bbfdef510c8c42806f373d05c8f.svg" align=middle width=28.401664499999992pt height=22.55708729999998pt/> (repulsion) term is in fact a broad-support, dense
convolution with the Student t-distribution kernel on non-equispaced,
scattered data points. As the matrix is dense, a naive method for
calculating the term takes <img src="svgs/3987120c67ed5a9162aa9841b531c3a9.svg" align=middle width=43.02219404999999pt height=26.76175259999998pt/> arithmetic operations. The quadratic
complexity limits the practical use of t-SNE to small graphs. Two types
of existing approaches reduce the quadratic complexity to <img src="svgs/ff514eba41c59f90c20d895e80719763.svg" align=middle width=72.2268393pt height=24.65753399999998pt/>,
they are typified by t-SNE-BH and FIt-SNE. The algorithm t-SNE-BH,
introduced by van der Maaten [[3](#VanDerMaaten2014)], is based on the
Barnes-Hut algorithm. The broad-support convolution is factored into
<img src="svgs/0d4b7f5b66e994af32a32cfa26868d53.svg" align=middle width=59.62030469999999pt height=24.65753399999998pt/> convolutions of narrow support, at multiple spatial levels,
each narrowly supported algorithm takes <img src="svgs/1f08ccc9cd7309ba1e756c3d9345ad9f.svg" align=middle width=35.64773519999999pt height=24.65753399999998pt/> operations. FIt-SNE,
presented by Linderman et al. [[4](#Linderman2019)], may be viewed as based
on non-uniform fast Fourier transforms. The execution time of each
approximate algorithm becomes dominated by the <img src="svgs/10113745b95b6c955f7c8b1959d3e01d.svg" align=middle width=27.12314054999999pt height=22.55708729999998pt/>
(attraction) term computation. The execution time also faces a steep
rise from 2D to 3D embedding.

### SG-t-SNE-Π

With the algorithm SG-t-SNE we extend the use of t-SNE to any sparse
stochastic graph <img src="svgs/1770ef1ded78d8145af3b517043e993d.svg" align=middle width=102.60478964999999pt height=24.65753399999998pt/>. The key input
is the stochastic matrix <img src="svgs/c4f6d7dc2dfb5ec67bc23d67c7fc2875.svg" align=middle width=73.75405784999998pt height=24.65753399999998pt/>,
<img src="svgs/c6a898940e1609c4b1528de9cfd03bf7.svg" align=middle width=80.90651249999999pt height=24.657735299999988pt/>, associated with the graph, where <img src="svgs/22ef4f9a3cc0b415bd468f738e64468f.svg" align=middle width=22.930086299999992pt height=14.15524440000002pt/> is not
restricted to the form of
(1).
We introduce a parametrized, non-linear rescaling mechanism to explore
the graph sparsity. We determine rescaling parameters <img src="svgs/9766609a48282e6f30837a712595b37c.svg" align=middle width=13.16154179999999pt height=14.15524440000002pt/> by


<p align="center"><img src="svgs/5c2a663f87e1bd635fb1636f3ec048bd.svg" align=middle width=296.7352509pt height=40.5367314pt/></p>


where <img src="svgs/f10fc2142d969a09dda2eea96cede2fb.svg" align=middle width=39.72592304999999pt height=22.831056599999986pt/> is an input parameter and <img src="svgs/f50853d41be7d55874e952eb0d80c53e.svg" align=middle width=9.794543549999991pt height=22.831056599999986pt/> is a monotonically
increasing function. We set <img src="svgs/0816c17505ab3faeea94b82afa429bbe.svg" align=middle width=63.28758314999998pt height=24.65753399999998pt/> in the present version of
SG-t-SNE-Π. Unlike
(2), the rescaling mechanism
(6) imposes no constraint on the graph,
its solution exists unconditionally. For the conventional t-SNE as a
special case, we set <img src="svgs/cd5494d95bfaee6295fdcf130852ed2d.svg" align=middle width=39.72592304999999pt height=22.831056599999986pt/> by default. One may still make use of
and exploit the benefit of rescaling (<img src="svgs/894e9a25f2969025d613371aa8b628c0.svg" align=middle width=39.72592304999999pt height=22.831056599999986pt/>).

With the implementation SG-t-SNE-Π, we accelerate the entire
gradient calculation of SG-t-SNE and enable practical 3D embedding of
large sparse graphs on modern desktop and laptop computers. We
accelerate the computation of both <img src="svgs/c6e87bbfdef510c8c42806f373d05c8f.svg" align=middle width=28.401664499999992pt height=22.55708729999998pt/> and <img src="svgs/10113745b95b6c955f7c8b1959d3e01d.svg" align=middle width=27.12314054999999pt height=22.55708729999998pt/> terms
by utilizing the matrix structures and the memory architecture in
tandem.

#### Accelerated accumulation of attractive interactions 

The matrix <img src="svgs/10113745b95b6c955f7c8b1959d3e01d.svg" align=middle width=27.12314054999999pt height=22.55708729999998pt/> in the attractive interaction term of
(5) has the same sparsity pattern as
matrix <img src="svgs/384591906555413c452c93e493b2d4ec.svg" align=middle width=12.92230829999999pt height=22.55708729999998pt/>, regardless of iterative changes in <img src="svgs/61ccc6d099c3b104d8de703a10b20230.svg" align=middle width=14.20083224999999pt height=22.55708729999998pt/>.
Sparsity patterns are generally irregular. Matrix-vector products with
irregular sparse matrix invoke irregular memory accesses and incur
non-equal, prolonged access latencies on hierarchical memories. We
moderate memory accesses by permuting the rows and columns of matrix
<img src="svgs/384591906555413c452c93e493b2d4ec.svg" align=middle width=12.92230829999999pt height=22.55708729999998pt/> such that rows and columns with similar nonzero patterns
are placed closer together. The permuted matrix becomes block-sparse
with denser blocks, resulting in better data locality in memory reads
and writes.

The permuted matrix <img src="svgs/384591906555413c452c93e493b2d4ec.svg" align=middle width=12.92230829999999pt height=22.55708729999998pt/> is stored in the Compressed Sparse
Blocks (CSB) storage format [[5](#Buluc2009)]. We utilize the CSB routines
for accessing the matrix and calculating the matrix-vector products with
the sparse matrix <img src="svgs/10113745b95b6c955f7c8b1959d3e01d.svg" align=middle width=27.12314054999999pt height=22.55708729999998pt/>. The elements of the <img src="svgs/10113745b95b6c955f7c8b1959d3e01d.svg" align=middle width=27.12314054999999pt height=22.55708729999998pt/>
matrix are formed on the fly during the calculation of the attractive
interaction term.

#### Accelerated accumulation of repulsive interactions 

We factor the convolution in the repulsive interaction term of
(5) into three consecutive convolutional
operations. We introduce an internal equispaced grid within the spatial
domain of the embedding points at each iteration, similar to the
approach used in FIt-SNE [[4](#Linderman2019)]. The three convolutional
operations are:

`S2G`: Local translation of the scattered (embedding) points to their
neighboring grid points.

`G2G`: Convolution across the grid with the same t-distribution kernel
function, which is symmetric, of broad support, and aperiodic.

`G2S`: Local translation of the gridded data to the scattered points.

The `G2S` operation is a gridded interpolation and `S2G` is its
transpose; the arithmetic complexity is <img src="svgs/0a84590e8a7ac80b1a5bbdeb714a6bf4.svg" align=middle width=55.52357579999999pt height=27.91243950000002pt/>, where <img src="svgs/31fae8b8b78ebe01cbfbe2fe53832624.svg" align=middle width=12.210846449999991pt height=14.15524440000002pt/> is the
interpolation window size per side. Convolution on the grid takes
<img src="svgs/ea79b06be39381b0524c4f8743c93d7f.svg" align=middle width=97.56834119999999pt height=24.65753399999998pt/> arithmetic operations, where <img src="svgs/423cdbe4a0e748ce336927701aea61dc.svg" align=middle width=16.69284539999999pt height=14.15524440000002pt/> is the
number of grid points, i.e., the grid size. The grid size is determined
by the range of the embedding points at each iteration, with respect to
the error tolerance set by default or specified by the user. In the
current implementation, the local interpolation method employed by
SG-t-SNE-Π is accurate up to cubic polynomials in <img src="svgs/2103f85b8b1477f430fc407cad462224.svg" align=middle width=8.55596444999999pt height=22.831056599999986pt/> separable
variables (<img src="svgs/3fa1f779de09763d248814c0c4f40d07.svg" align=middle width=69.74298869999998pt height=22.831056599999986pt/>).

Although the arithmetic complexity is substantially reduced in
comparison to the quadratic complexity of the direct way, the factored
operations suffer either from memory access latency or memory capacity
issues, which were not recognized or resolved in existing t-SNE
software. The scattered translation incurs high memory access latency.
The aperiodic convolution on the grid suffers from excessive use of
memory when the grid is periodically extended in all sides at once by
zero padding. The exponential memory growth with <img src="svgs/2103f85b8b1477f430fc407cad462224.svg" align=middle width=8.55596444999999pt height=22.831056599999986pt/> limits the
embedding dimension or the graph size.

We resolve these memory latency and capacity issues in SG-t-SNE-Π.
Prior to `S2G`, we relocate the scattered data points to the grid bins.
This binning process has two immediate benefits. It improves data
locality in the subsequent interpolation. It also establishes a data
partition for parallel, multi-threaded execution of the scattered
interpolation. We omit the parallelization details. For `G2G`, we
implement aperiodic convolution by operator splitting, without using
extra memory.

#### Rapid intra-term and inter-term data relocations 

In sparse or structured matrix computation of <img src="svgs/a36f952972b0f0fefd874eb09e26580e.svg" align=middle width=82.27261349999998pt height=24.65753399999998pt/> arithmetic
complexity, the execution time is dominated by memory accesses. We have
described
in the previous sections how we use
intra-term permutations to improve data locality and reduce memory
access latency in computing the [attraction](#accelerated-accumulation-of-attractive-interactions) and the [repulsion](#accelerated-accumulation-of-repulsive-interactions) terms of
(5). In addition, we permute and relocate
in memory the embedding data points between the two terms, at every
iteration step. The inter-term data relocation is carried out at
multiple layers, exploiting block-wise memory hierarchy. The data
permutation overhead is well paid-off by the much shortened time for
arithmetic calculation with the permuted data. We use Π in the
software name SG-t-SNE-Π to signify the importance and the role of
the permutations in accelerating t-SNE algorithms, including the
conventional one, and enabling 3D embeddings.

### Supplementary material 

Supplementary material and performance plots are found at
<http://t-sne-pi.cs.duke.edu>.


## References

[1] <a name="Pitsianis2019"></a> N. Pitsianis, A.-S. Iliopoulos, D. Floros, and
X. Sun. [Spaceland embedding of sparse stochastic
graphs](https://doi.org/10.1109/HPEC.2019.8916505). In *IEEE High Performance
Extreme Computing Conference*, 2019.

[2] <a name="Maaten2008"></a> L. van der Maaten and G. Hinton. [Visualizing data
using t-SNE](http://www.jmlr.org/papers/v9/vandermaaten08a.html). *Journal of
Machine Learning Research* 9(Nov):2579–2605, 2008.

[3] <a name="VanDerMaaten2014"></a> L. van der Maaten. [Accelerating t-SNE using
tree-based algorithms](http://jmlr.org/papers/v15/vandermaaten14a.html).
*Journal of Machine Learning Research* 15(Oct):3221–3245, 2014.

[4] <a name="Linderman2019"></a> G. C. Linderman, M. Rachh, J. G. Hoskins, S.
Steinerberger, and Y. Kluger. [Fast interpolation-based t-SNE for improved
visualization of single-cell RNA-seq
data](https://doi.org/10.1038/s41592-018-0308-4). *Nature Methods*
16(3):243–245, 2019.

[5] <a name="Buluc2009"></a> A. Buluç, J. T. Fineman, M. Frigo, J. R. Gilbert,
and C. E. Leiserson. [Parallel sparse matrix-vector and matrix-transpose-vector
multiplication using compressed sparse
blocks](https://doi.org/10.1145/1583991.1584053). In *Proceedings of Annual
Symposium on Parallelism in Algorithms and Architectures*, pp. 233–244, 2009.

[6] <a name="Pitsianis2019b"></a> N. Pitsianis, D. Floros, A.-S. Iliopoulos, and
X. Sun. [SG-t-SNE-Π: Swift neighbor embedding of sparse stochastic
graphs](https://doi.org/10.21105/joss.01577). *Journal of Open Source Software*
4(39):1577, 2019.


## Getting started 

### System environment 

SG-t-SNE-Π is developed for shared-memory computers with multi-threading,
running Linux or macOS operating system. The source code must be compiled with a
C++ compiler which supports Cilk. The current release is tested with
[OpenCilk](http://opencilk.org) 1.0 (based on LLVM/Tapir `clang++` 10.0.1) and
Intel Cilk Plus (GNU `g++` 7.4.0 and Intel `icpc` 19.0.4.233).

> WARNING: Intel Cilk Plus is deprecated and not supported in newer versions of
> GNU `g++` and Intel `icpc`.)

### Prerequisites 

SG-t-SNE-Π uses the following open-source software:

-   [FFTW3](http://www.fftw.org/) 3.3.8
-   [METIS](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview) 5.1.0
-   [FLANN](https://www.cs.ubc.ca/research/flann/) 1.9.1
-   [Intel TBB](https://01.org/tbb) 2019
-   [LZ4](https://github.com/lz4/lz4) 1.9.1
-   [Doxygen](http://www.doxygen.nl/) 1.8.14

On Ubuntu:

    sudo apt-get install libtbb-dev libflann-dev libmetis-dev libfftw3-dev liblz4-dev doxygen

On macOS:

    sudo port install flann tbb metis fftw-3 lz4 doxygen

### Installation 

#### Using configure + make 

##### Basic instructions 

To generate the SG-t-SNE-Π library, test and demo programs:

    ./configure
    make all

To specify the `C++` compiler:

    ./configure CXX=<compiler-executable>

To test whether the installation is successful:

    bin/test_modules

To generate the documentation:

    make documentation

##### Support of the conventional t-SNE 

SG-t-SNE-Π supports the conventional t-SNE algorithm, through a set
of preprocessing functions. Issue

    make tsnepi
    cp bin/tsnepi <BHTSNEPATH>/bh_tsne

to generate the `bin/tsnepi` binary, which is fully compatible with the
[existing wrappers](https://github.com/lvdmaaten/bhtsne/) provided by van der
Maaten [[3](#VanDerMaaten2014)], and replace the `bh_tsne`
binary. `<BHTSNEPATH>` is the installation path of
[`bhtsne`](https://github.com/lvdmaaten/bhtsne/).

##### MATLAB interface 

To compile the SG-t-SNE-Π MATLAB wrappers, use the
`--enable-matlab` option in the `configure` command. The default
MATLAB installation path is `/opt/local/matlab`; otherwise, set
`MATLABROOT`:

    ./configure --enable-matlab MATLABROOT=<matlab-path>

#### Using meson 

First, install the [Meson build system](https://mesonbuild.com/), for example
via the [Python Package Index (PyPI)](https://pypi.python.org/pypi/meson/):

    pip3 install --user meson

For more information and alternative methods for installing Meson, see [Getting
meson](https://mesonbuild.com/Getting-meson.html).

##### Basic instructions 

To configure the SG-t-SNE-Π library, demos, conventional t-SNE interface, and
documentation, issue the following:

    CXX=<compiler-executable> meson <build-options> <build-path>

This will create and configure the build directory at `<build-path>`.  The
`<build-options>` flags are optional.  For more information on the available
Meson build options, see [meson_options.txt](meson_options.txt).

To compile SG-t-SNE-Π within the build directory, issue:

    meson compile -C <build-path>

To install all relevant targets, issue:

    meson install -C <build-path>

By default, this will install targets in sub-directories `lib`, `bin`,
`include`, and `doc` within the SG-t-SNE-Π project directory.  To change the
installation prefix, specify the `-Dprefix=<install-prefix>` option when
configuring with Meson.

You may test the installation with:

    <install-prefix>/bin/test_modules

##### MATLAB interface 

To compile the SG-t-SNE-Π MATLAB wrappers, specify `-Denable_matlab=true` and
`-Dmatlabroot=<path-to-matlab-installation>` when configuring.

If building with a compiler which uses an Intel Cilk Plus implementation, you
may also need to set `-Ddir_libcilkrts=<path-to-libcilkrts.so-parent-dir>`.
This is not necessary when building with OpenCilk.

### Usage demo 

We provide two data sets of modest size for demonstrating stochastic
graph embedding with SG-t-SNE-Π:

    tar -xvzf data/mobius-graph.tar.gz
    bin/demo_stochastic_matrix mobius-graph.mtx

    tar -xvzf data/pbmc-graph.tar.gz
    bin/demo_stochastic_matrix pbmc-graph.mtx

The [MNIST data set](http://yann.lecun.com/exdb/mnist/) can be tested using [existing wrappers](https://github.com/lvdmaaten/bhtsne/) provided
by van der Maaten [[3](#VanDerMaaten2014)].

## License and community guidelines 

The SG-t-SNE-Π library is licensed under the [GNU general public
license v3.0](https://github.com/fcdimitr/sgtsnepi/blob/master/LICENSE).
To contribute to SG-t-SNE-Π or report any problem, follow our
[contribution
guidelines](https://github.com/fcdimitr/sgtsnepi/blob/master/CONTRIBUTING.md)
and [code of
conduct](https://github.com/fcdimitr/sgtsnepi/blob/master/CODE_OF_CONDUCT.md).

## Contributors 

*Design and development*:\
Nikos Pitsianis<sup>1,2</sup>, Dimitris Floros<sup>1</sup>,
Alexandros-Stavros Iliopoulos<sup>2</sup>, Xiaobai
Sun<sup>2</sup>

<sup>1</sup> Department of Electrical and Computer Engineering,
Aristotle University of Thessaloniki, Thessaloniki 54124, Greece\
<sup>2</sup> Department of Computer Science, Duke University, Durham, NC
27708, USA
