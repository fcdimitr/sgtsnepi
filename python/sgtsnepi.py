#!/usr/bin/env python
from julia import Main
from julia import Pkg

# load packages in Julia
Pkg.add( "SparseArrays" )
Pkg.add( "SGtSNEpi" )

from julia import SGtSNEpi

# prepare Julia wrappers
Main.eval("""
  using SparseArrays, SGtSNEpi
  import SGtSNEpi.sgtsnepi
  function scipyCSC_to_julia(A)
    m, n = A.shape
    colPtr = Int[i+1 for i in PyArray(A."indptr")]
    rowVal = Int[i+1 for i in PyArray(A."indices")]
    nzVal = Vector{Float64}(PyArray(A."data"))
    B = SparseMatrixCSC{Float64,Int}(m, n, colPtr, rowVal, nzVal)
  end
  sgtsnepi( A::PyObject ; kwargs... ) = sgtsnepi( scipyCSC_to_julia(A) ; kwargs... )
""")

# setup Python function

def sgtsnepi( A, **kwargs ):
    return SGtSNEpi.sgtsnepi( A, **kwargs )
