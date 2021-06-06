using LinearAlgebra, SparseArrays, MAT

# A = sprand(500,500,0.03); A = A + A'; A = A - spdiagm( 0=>diag(A) );

A = matread( "/tmp/mnist_hog-euclidean_k-25.mat" )["A"];

D = spdiagm( 0 => 1 ./ vec( sum(A;dims=1) ) );

P = A * D;

rows = Int32.( P.rowval .- 1 );
cols = Int32.( P.colptr .- 1 );
vals = P.nzval;

maxIter = 1000
earlyExag = 250

# See: https://stackoverflow.com/questions/33003174/calling-a-c-function-from-julia-and-passing-a-2d-array-as-a-pointer-of-pointers
timers = zeros( Float64, 6, maxIter );
ptr_timers = Ref{Ptr{Cdouble}}([Ref(timers,i) for i=1:size(timers,1):length(timers)]);

grid_sizes = zeros( Int32, maxIter );

# lib = libsgtsnepi  # [if JLL is available]
lib = "./build/libsgtsnepi.dylib"

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
                  2, 1.0, maxIter, earlyExag,
                  Int32.( size(P,1) ) )

Y = permutedims( unsafe_wrap( Array, dotsne(), (2, size(P,1)) ) )


timers2d = copy( timers )
Y2d = copy( Y )

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



using CairoMakie, Colors, GLMakie

GLMakie.activate!()

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

f = Figure()
axs = [Axis(f[i+1, 1], title = "$(i+1)-D") for i in 1:2]

vec_colors = distinguishable_colors(size(timers2d, 1))
empty!(axs[1])
tsum = zeros( maxIter )
for i = 1:size( timers2d, 1 )
  band!(axs[1], 1:maxIter, tsum, tsum + vec( timers2d[i,:] ),
        label = str_module[i],
        color = vec_colors[i] )
  tsum .+= timers2d[i,:]
end

empty!(axs[2])
tsum = zeros( maxIter )
b = []
for i = 1:size( timers3d, 1 )
  push!(b, band!(axs[2], 1:maxIter, tsum, tsum + vec( timers3d[i,:] ),
                 label = str_module[i],
                 color = vec_colors[i] ) )
  tsum .+= timers3d[i,:]
end

Legend(f[1,1], b, str_module,
       orientation = :horizontal, tellwidth = false, tellheight = true)

supertitle = f[0, :] = Label(
  f, "MNIST -- profiling results",
  tellwidth = false)

save("/tmp/mnist-plot.png", f, px_per_unit = 2)
