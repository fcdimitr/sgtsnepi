using LinearAlgebra, SparseArrays, MAT

# A = sprand(500,500,0.03); A = A + A'; A = A - spdiagm( 0=>diag(A) );

A = matread( "/tmp/mnist_hog-euclidean_k-25.mat" )["A"];

maxIter = 1000
earlyExag = 250

lib = "./build/libsgtsnepi.dylib"

function dotsne(A, d_Y, maxIter, earlyExag)


  D = spdiagm( 0 => 1 ./ vec( sum(A;dims=1) ) );

  P = A * D;

  rows = Int32.( P.rowval .- 1 );
  cols = Int32.( P.colptr .- 1 );
  vals = P.nzval;

  # See: https://stackoverflow.com/questions/33003174/calling-a-c-function-from-julia-and-passing-a-2d-array-as-a-pointer-of-pointers
  timers = zeros( Float64, 6, maxIter );
  ptr_timers = Ref{Ptr{Cdouble}}([Ref(timers,i) for i=1:size(timers,1):length(timers)]);

  grid_sizes = zeros( Int32, maxIter );

  # lib = libsgtsnepi  # [if JLL is available]

  dotsne() = ccall( (:tsnepi_c, lib ), Ptr{Cdouble},
                  ( Ptr{Ptr{Cdouble}}, Ptr{Cint},
                    Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble},
                    Ptr{Cdouble},
                    Cint,
                    Cint, Cdouble, Cint, Cint,
                    Cint ),
                  ptr_timers, grid_sizes,
                  rows, cols, vals,
                  C_NULL,
                  Int32.( nnz(P) ),
                  d_Y, 1.0, maxIter, earlyExag,
                  Int32.( size(P,1) ) )

  Y = permutedims( unsafe_wrap( Array, dotsne(), (2, size(P,1)) ) )

  Y, timers, grid_sizes

end


Y1d, timers1d, grid_size_1d = dotsne( A, 1, maxIter, earlyExag );
Y2d, timers2d, grid_size_2d = dotsne( A, 2, maxIter, earlyExag );
Y3d, timers3d, grid_size_3d = dotsne( A, 3, maxIter, earlyExag );


return




timers = zeros( Float64, 6, maxIter );
ptr_timers = Ref{Ptr{Cdouble}}([Ref(timers,i) for i=1:size(timers,1):length(timers)]);

dotsne() = ccall( (:tsnepi_c, lib ), Ptr{Cdouble},
                  (Ptr{Ptr{Cdouble}}, Ptr{Cint}, Ptr{Cint},
                   Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint,
                   Cdouble, Cint, Cint, Cint),
                  ptr_timers, rows, cols, vals, C_NULL, Int32.( nnz(P) ),
                  3, 1.0, maxIter, earlyExag, Int32.( size(P,1) ) )

Y = permutedims( unsafe_wrap( Array, dotsne(), (3, size(P,1)) ) )

timers3d = copy( timers )
Y3d = copy( Y )



return

# ==================== INTERACTIVE PLOTS ====================

using Colors, GLMakie

f = Figure()
ax1 = Axis(f[1, 1])
ax2 = Axis(f[1, 1], yticklabelcolor = :blue, yaxisposition = :right)

hidespines!(ax2)
hidexdecorations!(ax2)

vec_colors = distinguishable_colors(size(timers2d, 1))
tsum = zeros( maxIter )
for i = 1:size( timers2d, 1 )
  band!(ax1, 1:maxIter, tsum, tsum + vec( timers2d[i,:] ),
        label = str_module[i],
        color = vec_colors[i] )
  tsum .+= timers2d[i,:]
end

lines!(ax2, 1..maxIter, grid_sizes, color = :blue, linewidth = 3)

str_module = [
  "PQ", "Gridding", "S2G", "G2G", "G2S", "F&Z"
]

axislegend(ax1, str_module)


return

# ==================== NON-INTERACTIVE (USE ON SERVERS) ====================

using CairoMakie, Colors

f = Figure( resolution = (1000, 1200) )
axs       = [Axis(f[i+1, 1], title = "$(i)-D") for i in 1:3]
axs_right = [Axis(f[i+1, 1], title = "$(i)-D",
                  yticklabelcolor = :blue,
                  yaxisposition = :right) for i in 1:3]

vec_colors = distinguishable_colors(size(timers1d, 1));

empty!(axs[1])
tsum = zeros( maxIter );
for i = 1:size( timers1d, 1 )
  band!(axs[1], 1:maxIter, tsum, tsum + vec( timers1d[i,:] ),
        label = str_module[i],
        color = vec_colors[i] );
  tsum .+= timers1d[i,:];
end

lines!(axs_right[1], 1..maxIter, grid_size_1d, color = :blue, linewidth = 3)

empty!(axs[2])
tsum = zeros( maxIter );
for i = 1:size( timers2d, 1 )
  band!(axs[2], 1:maxIter, tsum, tsum + vec( timers2d[i,:] ),
        label = str_module[i],
        color = vec_colors[i] );
  tsum .+= timers2d[i,:];
end

lines!(axs_right[2], 1..maxIter, grid_size_2d, color = :blue, linewidth = 3)

empty!(axs[3])
tsum = zeros( maxIter );
b = []
for i = 1:size( timers3d, 1 )
  push!(b, band!(axs[3], 1:maxIter, tsum, tsum + vec( timers3d[i,:] ),
                 label = str_module[i],
                 color = vec_colors[i] ) );
  tsum .+= timers3d[i,:];
end

lines!(axs_right[3], 1..maxIter, grid_size_3d, color = :blue, linewidth = 3)

Legend(f[1,1], b, str_module,
       orientation = :horizontal, tellwidth = false, tellheight = true)

supertitle = f[0, :] = Label(
  f, "MNIST -- profiling results",
  tellwidth = false)

save("/tmp/mnist-plot.png", f, px_per_unit = 1)
