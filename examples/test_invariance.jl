using LinearAlgebra
using StatsBase
using Flux
using CUDA

include("../src/layers.jl")
include("../src/rotational_utils.jl")

batch_size = 32
frames = 64
dim = 768

si = Float32.(randn(dim,frames,batch_size)) |> gpu

T_loc = (get_rotation(frames,batch_size), get_translation(frames,batch_size)) |> gpu

# Get 1 global SE(3) transformation for each batch.
T_glob = (get_rotation(batch_size), get_translation(batch_size)) 

T_glob = (stack([T_glob[1] for i in 1:frames],dims = 3), stack([T_glob[2] for i in 1:frames],dims = 3)) |> gpu
T_new = T_T(T_glob,T_loc)

ipa = IPAStructureModuleLayer(IPA_settings(dim)) |> gpu

T_loc, si_loc = ipa(T_loc,si)
T_glob, si_glob = ipa(T_new, si)

invariance_error = maximum(abs.(si_glob .- si_loc)) 

@show invariance_error # ≈ 10^-6