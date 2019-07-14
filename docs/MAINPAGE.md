# sgtsnepi: Swift Neighbor Embedding of Sparse Stochastic Graphs

## Getting started 

### System environment 

SG-t-SNE-Π is developed for shared-memory computers with
multi-threading, running Linux or macOS operating system. The source
code is (to be) compiled by a `C++` compiler supporting Cilk. The
current release is tested with the `GNU g++` compiler 7.4.0 and the
`Intel` `icpc` compiler 19.0.4.233.

### Prerequisites 

SG-t-SNE-Π uses the following open-source software:

-   `FFTW3` 3.3.8

-   `METIS` 5.1.0

-   `FLANN` 1.9.1

-   `Intel TBB` 2019

-   `Doxygen` 1.8.14

On `Ubuntu`:

    sudo apt-get install libtbb-dev libflann-dev libmetis-dev libfftw3-dev doxygen

On `macOS`:

    sudo port install flann tbb metis fftw-3

### Installation 

#### Basic instructions 

To generate the SG-t-SNE-Π library, test and demo programs:

    ./configure
    make all

To specify the `C++` compiler:

    ./configure CXX=<compiler-executable>

To test whether the installation is successful:

    bin/test_modules

To generate the documentation:

    make documentation

#### Support of the conventional t-SNE 

SG-t-SNE-Π supports the conventional t-SNE algorithm, through a set
of preprocessing functions. Issue

    make tsnepi

to generate the `bin/tsnepi` binary, which is fully compatible with the
[existing wrappers](https://github.com/lvdmaaten/bhtsne/) provided by van der Maaten [[6](#VanDerMaaten2014)].

#### MATLAB interface 

To compile the SG-t-SNE-Π `MATLAB` wrappers, use the
`--enable-matlab` option in the `configure` command. The default
`MATLAB` installation path is `/opt/local/matlab`; otherwise, set
`MATLABROOT`:

    ./configure --enable-matlab MATLABROOT=<matlab-path>

### Usage demo 

We provide two data sets of modest size for demonstrating stochastic
graph embedding with SG-t-SNE-Π:

    tar -xvzf data/mobius-graph.tar.gz
    bin/demo_stochastic_matrix mobius-graph.mtx

    tar -xvzf data/pbmc-graph.tar.gz
    bin/demo_stochastic_matrix pbmc-graph.mtx

The [MNIST data set](http://yann.lecun.com/exdb/mnist/) can be tested using [existing wrappers](https://github.com/lvdmaaten/bhtsne/) provided
by van der Maaten [[6](#VanDerMaaten2014)].
