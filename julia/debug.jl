using LinearAlgebra, FFTW

n  = 10000
d  = 3
np = 1
nGrid = 140;

# random data
Y = rand(d, n);
Y .-= minimum( Y; dims = 2 );

h = 0.2

function s2g!( VGrid, VScat, Y, np )

  (d, n) = size( Y )
  nGridDim = size(VGrid, 1)
  m = d+1

  if d == 1
    ccall( (:s2g1d, "../build/libsgtsnepi.dylib"), Cvoid,
           (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
            Cint, Cint, Cint, Cint, Cint),
           VGrid, Y, VScat, nGridDim, np, n, d, m )
  elseif d == 2
    ccall( (:s2g2d, "../build/libsgtsnepi.dylib"), Cvoid,
           (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
            Cint, Cint, Cint, Cint, Cint),
           VGrid, Y, VScat, nGridDim, np, n, d, m )
  else
    ccall( (:s2g3d, "../build/libsgtsnepi.dylib"), Cvoid,
           (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
            Cint, Cint, Cint, Cint, Cint),
           VGrid, Y, VScat, nGridDim, np, n, d, m )
  end

end

function g2s( PhiGrid, Y )

  (d, n) = size(Y)
  nGridDim = size(PhiGrid, 1)
  m  = d+1
  PhiScat = zeros( d+1, n );

  if d == 1
    ccall( (:g2s1d, "../build/libsgtsnepi.dylib"), Cvoid,
           (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
            Cint, Cint, Cint, Cint),
           PhiScat, PhiGrid, Y, nGridDim, n, d, m )
  elseif d == 2
    ccall( (:g2s2d, "../build/libsgtsnepi.dylib"), Cvoid,
           (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
            Cint, Cint, Cint, Cint),
           PhiScat, PhiGrid, Y, nGridDim, n, d, m )
  else
    ccall( (:g2s3d, "../build/libsgtsnepi.dylib"), Cvoid,
           (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
            Cint, Cint, Cint, Cint),
           PhiScat, PhiGrid, Y, nGridDim, n, d, m )
  end

  PhiScat

end

function nuconv( y_arg, np, nGridDim )
  y_all = copy(y_arg)
  # y_in, p_in, y_ex, p_ex = separate_points(y_all, 0.2);

  (d, n) = size(y_all)

  # ========== INTERNAL POINTS
  VScat    = [ones(1,n); y_all];
  m = d+1

  maxy = maximum(y_all)
  y_all ./= maxy
  y_all[y_all.==1.0] .-= eps()
  y_all .*= (nGridDim-1)

  h = maxy / (nGridDim-1-eps())

  szV = vcat( repeat( [nGridDim+2], d ), m )

  VGrid = zeros( szV...,np )

  s2g!( VGrid, VScat, y_all, np )

  (d, n) = size(y_all)
  @show size(VGrid)
  VGrid = dropdims( sum( VGrid; dims = d+2 ); dims = d+2 )
  @show size(VGrid)
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
    timers = zeros( 14 )
    @time ccall( (:conv3dnopad, "../build/libsgtsnepi.dylib"), Cvoid,
           (Ptr{Cdouble}, Ptr{Cdouble},
            Cdouble, Ptr{Cint}, Cint, Cint, Cint, Ptr{Cdouble}),
           PhiGrid, VGrid, h, nGridDims, m, d, np, timers )
  end

  @time R = convFFT( VGrid, h )[1]
  @show norm( R - PhiGrid )

  # PhiScat = zeros( d+1, size(y_all,2) );
  PhiScat = g2s( PhiGrid, y_all )

  PhiScat

end

function convFFT(VGrid::Array{Float64,2}, h)

  ng   = [size(VGrid,1) size(VGrid,2) 1]
  nDim = 1
  nVec = ng[nDim+1]

  nFFT = 0

  # first column of Toeplitz matrix
  t = 1 ./ ( 1 .+ h^2 * (0:1:ng[1]-1).^2 ).^2

  # kernel even-odd freq. (padded)
  dPad = rfft( [t; 0; t[ng[1]:-1:2]], 1 )
  nFFT = nFFT + size(t, 2)

  # signal even-odd freq. (0-padded)
  yPad = rfft( [VGrid; zeros(ng[1],2)], 1 )
  nFFT = nFFT + size(VGrid, 2)

  # convolution
  PhiGrid = irfft( dPad .* yPad, 2*ng[1], 1 )
  nFFT = nFFT + size(PhiGrid, 2)

  # drop padded outputs
  PhiGrid = PhiGrid[ 1:ng[1], : ]

  return [PhiGrid,nFFT]

end


function convFFT(VGrid::Array{Float64,3}, h)

  ng   = [size(VGrid,1) size(VGrid,2) size(VGrid,3) 1]
  nDim = 2
  nVec = ng[nDim+1]

  nFFT = 0

  # construct 2-D kernel matrix
  T = 1 ./ ( 1 .+ h^2 .* ( ((0:1:ng[1]-1).^2)' .+ ((0:1:ng[2]-1).^2) ) ).^2

  # kernel frequencies [padded]
  T    = [ T; zeros(1, size(T,2)); T[ng[1]:-1:2,:] ]
  T    = [ T  zeros(size(T,1), 1)  T[:,ng[2]:-1:2] ]
  Dpad = fft( T )

  nFFT = nFFT + size( T, 3 )

  # signal frequencies [0-padded]
  Ypad = fft( [VGrid zeros(ng[1],ng[2],3); zeros(ng[1],2*ng[2],3)], [1,2] )

  nFFT = nFFT + size( VGrid, 3 )

  # convolution
  PhiGrid = ifft( Dpad .* Ypad, [1,2] )
  PhiGrid = PhiGrid[ 1:ng[1], 1:ng[2], : ]

  nFFT = nFFT + size( PhiGrid, 3 )

  return [PhiGrid,nFFT]

end

function convFFT(VGrid::Array{Float64,4}, h)
  @time begin
  ng   = [size(VGrid)... 1]
  nDim = length(ng) - 2
  nVec = ng[nDim+1]

  nFFT = 0

  # construct 2-D kernel matrix
  T = 1 ./ ( 1 .+
             h^2 * (  ( (0:1:ng[1]-1).^2)' .+
                      ( (0:1:ng[2]-1).^2)  .+
                      reshape( (0:1:ng[3]-1).^2, (1, 1, ng[3]) ) ) ).^2

  # kernel frequencies [padded]
  T = cat( T, zeros(1, size(T,2), size(T,3) ), T[ng[1]:-1:2,:,:]; dims = 1 )
  T = cat( T, zeros(size(T,1), 1, size(T,3) ), T[:,ng[2]:-1:2,:]; dims = 2 )
  T = cat( T, zeros(size(T,1), size(T,2), 1 ), T[:,:,ng[3]:-1:2]; dims = 3 )
  end

  @time begin
  P = plan_rfft( T )

  Dpad = P * T

  nFFT = nFFT + 1

  end

  @time begin
    Vpad = zeros( (ng[1:3].*2)..., 4 )
    Vpad[1:ng[1], 1:ng[2], 1:ng[3], :] .= VGrid
  end

  @time begin
  P = plan_rfft( Vpad, [1,2,3] )
  Ypad = P * Vpad
  end

  @time begin
    R = plan_irfft( Ypad, 2*ng[1], [1,2,3] )
  end

  @time begin
    Ypad .*= Dpad
  end
  @time begin
  PhiGrid = R * Ypad
  end

  PhiGrid = PhiGrid[ 1:ng[1], 1:ng[2], 1:ng[3], : ]
  return [PhiGrid,nFFT]

end
