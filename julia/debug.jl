
n  = 100
d  = 2
np = 1
nGrid = 14;

# random data
Y = rand(d, n);
Y .-= minimum( Y; dims = 2 );

h = 0.2

function _qq_interp_c( X::DenseMatrix, h::Real, np::Int )

  frep = zeros( size(X) )
  timers = zeros( 5 )

  zeta = ccall( ( :computeFrepulsive_interp, "../build/libsgtsnepi.dylib" ),
                Cdouble,
                ( Ptr{Cdouble},
                  Ptr{Cdouble},
                  Cint, Cint,
                  Cdouble, Cint, Ptr{Cdouble}),
                frep, X,
                size(X,2), size(X,1),
                h, np, timers )

  frep, zeta

end

function nuconv( PhiScat, y_in, VScat, n, d, m, np, nGridDim )

  y = copy(y_in)
  maxy = maximum(y)
  y ./= maxy
  y[y.==1.0] .-= eps()
  y .*= (nGridDim-1)

  h = maxy / (nGridDim-1-eps())

  szV = vcat( repeat( [nGridDim+2], d ), m )
  # szV = m * ( ( nGridDim + 2 ) ^ d )

  VGrid = zeros( szV...,np )

  if d == 1
    ccall( (:s2g1d, "../build/libsgtsnepi.dylib"), Cvoid,
           (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
            Cint, Cint, Cint, Cint, Cint),
           VGrid, y, VScat, nGridDim+2, np, n, d, m )
  elseif d == 2
    ccall( (:s2g2d, "../build/libsgtsnepi.dylib"), Cvoid,
           (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
            Cint, Cint, Cint, Cint, Cint),
           VGrid, y, VScat, nGridDim+2, np, n, d, m )
  else
    ccall( (:s2g3d, "../build/libsgtsnepi.dylib"), Cvoid,
           (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
            Cint, Cint, Cint, Cint, Cint),
           VGrid, y, VScat, nGridDim+2, np, n, d, m )
  end

  sum( VGrid; dims = d+1 )

  PhiGrid = zeros( szV... )
  nGridDims = UInt32.( fill(nGridDim+2, d) )

  if d == 1
    ccall( (:conv1dnopad, "../build/libsgtsnepi.dylib"), Cvoid,
           (Ptr{Cdouble}, Ptr{Cdouble},
            Cdouble, Ptr{Cint}, Cint, Cint, Cint),
           PhiGrid, VGrid, h, nGridDims, m, d, np )
  elseif d == 2
    ccall( (:conv2dnopad, "../build/libsgtsnepi.dylib"), Cvoid,
           (Ptr{Cdouble}, Ptr{Cdouble},
            Cdouble, Ptr{Cint}, Cint, Cint, Cint),
           PhiGrid, VGrid, h, nGridDims, m, d, np )
  else
    ccall( (:conv3dnopad, "../build/libsgtsnepi.dylib"), Cvoid,
           (Ptr{Cdouble}, Ptr{Cdouble},
            Cdouble, Ptr{Cint}, Cint, Cint, Cint),
           PhiGrid, VGrid, h, nGridDims, m, d, np )
  end

  if d == 1
    ccall( (:g2s1d, "../build/libsgtsnepi.dylib"), Cvoid,
           (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
            Cint, Cint, Cint, Cint),
           PhiScat, PhiGrid, y, nGridDim+2, n, d, m )
  elseif d == 2
    ccall( (:g2s2d, "../build/libsgtsnepi.dylib"), Cvoid,
           (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
            Cint, Cint, Cint, Cint),
           PhiScat, PhiGrid, y, nGridDim+2, n, d, m )
  else
    ccall( (:g2s3d, "../build/libsgtsnepi.dylib"), Cvoid,
           (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
            Cint, Cint, Cint, Cint),
           PhiScat, PhiGrid, y, nGridDim+2, n, d, m )
  end

  PhiGrid

end


f_g, z_g = _qq_interp_c( Y, h, np );

VScat = [ones(1,n); Y];
PhiScat = zeros( d+1, n );
Yt = copy( Y )
ccall( (:nuconv, "../build/libsgtsnepi.dylib"), Cvoid,
       (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
        Ptr{Cint}, Ptr{Cint},
        Cint, Cint, Cint, Cint, Cint, Ptr{Cdouble} ),
       PhiScat, Yt, VScat,
       C_NULL, C_NULL,
       n, d, d+1, np, nGrid, C_NULL )

Ysq = sum( Y.^2; dims = 1 );
Z  = - sum( 2 * PhiScat[2:end,:] .* Y )
Z +=   sum( ( 1 .+ (2 .* Ysq) ) .* PhiScat[1,:]' )
Z -= n

f = ( (Y .* PhiScat[1,:]') - PhiScat[2:end,:] ) / Z

PhiScat2 = zeros( d+1, n );
VScat2   = [ones(1,n); Y];
PhiGrid  = nuconv( PhiScat2, Y, VScat2, n, d, d+1, np, nGrid )

Ysq = sum( Y.^2; dims = 1 );
Z  = - sum( 2 * PhiScat2[2:end,:] .* Y )
Z +=   sum( ( 1 .+ (2 .* Ysq) ) .* PhiScat2[1,:]' )
Z -= n

f = ( (Y .* PhiScat2[1,:]') - PhiScat2[2:end,:] ) / Z

f ≈ f_g
Z ≈ z_g
PhiScat2 ≈ PhiScat
