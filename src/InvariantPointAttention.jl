module InvariantPointAttention

using LinearAlgebra
using Flux
using ChainRulesCore
using BatchedTransformations
using UnPack

include("utils.jl")

include("softmax1.jl")
export softmax1

include("layers.jl")
export IPAConfig
export IPA
export StructureModule

end
