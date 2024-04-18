"""
softmax1(x, dims = 1)

Behaves like softmax, but as though there was an additional logit of zero along dims (which is excluded from the output). So the values will sum to a value between zero and 1.
"""
function softmax1(x::AbstractArray{T}; dims = 1) where {T}
    _zero = T(0)
    max_ = max.(fast_maximum2(x; dims), _zero)
    @fastmath out = exp.(x .- max_)
    tmp = sum(out, dims = dims)
    out ./ (tmp + exp.(-max_))
end
# Pirated/adapted from NNlib
fast_maximum2(x::AbstractArray{T}; dims) where {T} = @fastmath reduce(max, x; dims, init = float(T)(-Inf))

function ∇softmax1_data(dy::AbstractArray{T}, y::AbstractArray{S}; dims = 1) where {T,S}
    dx = if NNlib.within_gradient(y)
        tmp = dy .* y
        tmp .- y .* sum(tmp; dims)
    else
        # This path is faster, only safe for 1st derivatives though.
        # Was previously `∇softmax!(dx, dy, x, y; dims)` to allow CUDA overloads,
        # but that was slow: https://github.com/FluxML/NNlibCUDA.jl/issues/30
        out = similar(y, promote_type(T,S))  # sure to be mutable
        out .= dy .* y
        out .= out .- y .* sum(out; dims)
    end
end

function ChainRulesCore.rrule(::typeof(softmax1), x; dims = 1)
    y = softmax1(x; dims)
    softmax_pullback(dy) = (NoTangent(), ∇softmax1_data(unthunk(dy), y; dims))
    return y, softmax_pullback
end

"""
Projects the frame embedding => 6, and uses this to transform the input frames.
"""
struct BackboneUpdate
    layers::NamedTuple
end
Flux.@functor BackboneUpdate
function BackboneUpdate(s_dim::Int)
    layers = (
        linear = Dense(s_dim => 6), 
    )
    return BackboneUpdate(layers)
end
function (backboneupdate::BackboneUpdate)(Ti, si)
    bu = backboneupdate.layers.linear(si)
    arr = reshape(bu,3,2,size(si,2),:) 
    T = update_frame(Ti, arr)
    return T
end

"""
Returns a tuple of the IPA settings, with defaults for everything except dims. This can be passed to the IPA and IPCrossAStructureModuleLayer.
"""
IPA_settings(
    dims;
    c = 16,
    N_head = 12,
    N_query_points = 4,
    N_point_values = 8,
    c_z = 0,
    Typ = Float32,
    use_softmax1 = false
) = (
    dims = dims,
    c = c,
    N_head = N_head,
    N_query_points = N_query_points,
    N_point_values = N_point_values,
    c_z = c_z,
    Typ = Typ,
    pairwise = c_z > 0,
    use_softmax1 = use_softmax1
)


"""
Invariant Point Cross Attention (IPCrossA). Information flows from L (Keys, Values) to R (Queries).
"""
struct IPCrossA
    settings::NamedTuple
    layers::NamedTuple
end


Flux.@functor IPCrossA # provides parameter collection, gpu movement and more

function IPCrossA(settings::NamedTuple)
    dims, c, N_head, N_query_points, N_point_values, c_z, Typ, pairwise = settings 
    # Needs a slighyly unusual initialization - hat-tip: Kenta
    init = Flux.kaiming_uniform(gain = 1.0)
    if pairwise
        pair = Dense(c_z => N_head, bias = false; init)
        ipa_linear = Dense(N_head*c_z + N_head*c + 4*N_head*N_point_values => dims)
    else
        pair = nothing
        ipa_linear = Dense(N_head*c + 4*N_head*N_point_values => dims)
    end
    layers = (
            proj_qh = Dense(dims => c*N_head, bias = false; init),
            proj_kh = Dense(dims => c*N_head, bias = false; init),
            proj_vh = Dense(dims => c*N_head, bias = false; init),
            proj_qhp = Dense(dims => 3*N_head*N_query_points, bias = false; init),
            proj_khp = Dense(dims => 3*N_head*N_query_points, bias = false; init),
            proj_vhp = Dense(dims => 3*N_head*N_point_values, bias = false; init),
            ipa_linear = ipa_linear,
            pair = pair,
            gamma_h = min.(ones(Typ, N_head) .* Typ(0.541),1f2)
            )
    
    return IPCrossA(settings, layers)
end



"""
Strictly Self-IPA initialization
"""
# We could skip making this struct and just have it be a cross IPA struct. 
struct IPA
    settings::NamedTuple
    layers::NamedTuple
end

Flux.@functor IPA

function IPA(settings::NamedTuple)
    crossL = IPCrossA(settings)
    return IPA(crossL.settings, crossL.layers)
end

"""
Self-IPA can be run from both IPA and cross IPA, allowing for flexibility. Simply calls cross IPA on itself. 
"""
function (ipa::Union{IPA, IPCrossA})(T::Tuple{AbstractArray,AbstractArray}, S::AbstractArray; Z = nothing, mask = 0)
    return ipa(T, S, T, S; zij = Z, mask = mask)
end


#Attention props from L (Keys, Values) to R (Queries).
#Because IPA uses Q'K, our pairwise matrices are R-by-L
function (ipa::Union{IPCrossA, IPA})(TiL::Tuple{AbstractArray,AbstractArray}, siL::AbstractArray, TiR::Tuple{AbstractArray,AbstractArray}, siR::AbstractArray; zij = nothing, mask = 0)
   
    if zij != nothing
        #This is assuming the dims of zij are c, N_frames_L, N_frames_R, batch
        @assert size(zij,2) == size(siR,2)
        @assert size(zij,3) == size(siL,2) 
    end
    if mask != 0
        @assert size(mask,1) == size(siR, 2)
        @assert size(mask,2) == size(siL, 2)
    end
    
    # Get relevant parameters from our ipa struct.
    l = ipa.layers
    dims, c, N_head, N_query_points, N_point_values, c_z, Typ, pairwise = ipa.settings 
    if haskey(ipa.settings, :use_softmax1) #For compat
        use_softmax1 = ipa.settings.use_softmax1
    else
        use_softmax1 = false
    end
    
    rot_TiL, translate_TiL = TiL
    rot_TiR, translate_TiR = TiR
    
    N_frames_L = size(siL,2)
    N_frames_R = size(siR,2)

    gamma_h = softplus(clamp.(l.gamma_h,Typ(-100), Typ(100))) #Clamping

    w_C = Typ(sqrt(2/(9*N_query_points)))
    dim_scale = Typ(1/sqrt(c))

    qh = reshape(l.proj_qh(siR),(c,N_head,N_frames_R,:))
    kh = reshape(l.proj_kh(siL),(c,N_head,N_frames_L,:))
    vh = reshape(l.proj_vh(siL),(c,N_head,N_frames_L,:))
    qhp = reshape(l.proj_qhp(siR),(3,N_head*N_query_points,N_frames_R,:))
    khp = reshape(l.proj_khp(siL),(3,N_head*N_query_points,N_frames_L,:))
    vhp = reshape(l.proj_vhp(siL),(3,N_head*N_point_values,N_frames_L,:))

    # This should be Q'K, following IPA, which isn't like the regular QK'
    # Dot products between queries and keys.
                        #FramesR, c, N_head, Batch
    qhT = permutedims(qh, (3, 1, 2, 4))
                         #c, FramesL, N_head, Batch
    kh = permutedims(kh, (1, 3, 2, 4))
    qhTkh = permutedims(#FramesR, #FramesL, N_head, Batch
                        batched_mul(qhT,kh)
                        #N_head, FramesR, FramesL, Batch when we use (3,1,2,4)
                            ,(3,1,2,4))
    
    # Applying our transformations to the queries, keys, and values to put them in the global frame.
    Tqhp = reshape(T_R3(qhp, rot_TiR,translate_TiR),3,N_head,N_query_points,N_frames_R,:) 
    Tkhp = reshape(T_R3(khp, rot_TiL,translate_TiL),3,N_head,N_query_points,N_frames_L,:)
    Tvhp = T_R3(vhp, rot_TiL, translate_TiL)

    diffs_glob = unsqueeze(Tqhp, dims = 5) .- unsqueeze(Tkhp, dims = 4)
    sum_norms_glob = reshape(sum(diffs_glob.^2, dims = [1,3]),N_head,N_frames_R,N_frames_L,:) #Sum over points for each head
    

    att_arg = reshape(dim_scale .* qhTkh .- w_C/2 .* gamma_h .* sum_norms_glob,(N_head,N_frames_R,N_frames_L, :))
    
    if pairwise
        w_L = Typ(sqrt(1/3))
        bij = reshape(l.pair(zij),(N_head,N_frames_R,N_frames_L,:))
    else
        w_L = Typ(sqrt(1/2))
        bij = Typ(0)
    end

    # Setting mask to the correct dim for broadcasting. 
    if mask != 0 
        mask = unsqueeze(mask, dims = 1) 
    end

    if use_softmax1
        att = softmax1(w_L .* (att_arg .+ bij) .+ mask, dims = 3)
    else
        att = Flux.softmax(w_L .* (att_arg .+ bij) .+ mask, dims = 3)
    end

    # Applying the attention weights to the values.
    broadcast_att_oh = reshape(att,(1,N_head,N_frames_R,N_frames_L,:))
    broadcast_vh = reshape(vh, (c,N_head,1,N_frames_L,:))
    oh = reshape(sum(broadcast_att_oh .* broadcast_vh,dims = 4), c,N_head,N_frames_R,:)

    broadcast_att_ohp = reshape(att,(1,N_head,1,N_frames_R,N_frames_L,:))
    broadcast_tvhp = reshape(Tvhp,(3,N_head,N_point_values,1,N_frames_L,:))

    if use_softmax1
        pre_ohp_r = sum(broadcast_att_ohp.*broadcast_tvhp,dims=5)
        unreshaped_ohp_r = pre_ohp_r .+ (1 .- sum(broadcast_att_ohp, dims = 5)) .* reshape(translate_TiR, 3, 1, 1, N_frames_R, 1, :)
        ohp_r = reshape(unreshaped_ohp_r,3,N_head*N_point_values,N_frames_R,:)
    else
        ohp_r = reshape(sum(broadcast_att_ohp.*broadcast_tvhp,dims=5),3,N_head*N_point_values,N_frames_R,:)
    end

    #ohp_r were in the global frame, so we put those back in the recipient local
    ohp = T_R3_inv(ohp_r, rot_TiR, translate_TiR) 
    normed_ohp = sqrt.(sum(ohp.^2,dims = 1) .+ Typ(0.000001f0)) #Adding eps

    catty = vcat(
        reshape(oh, N_head*c, N_frames_R,:),
        reshape(ohp, 3*N_head*N_point_values, N_frames_R,:),
        reshape(normed_ohp, N_head*N_point_values, N_frames_R,:)
        ) 
    
    if pairwise
        broadcast_zij = reshape(zij,(c_z,1,N_frames_R,N_frames_L,:))
        broadcast_att_zij = reshape(att,(1,N_head,N_frames_R,N_frames_L,:))
        obh = sum(broadcast_zij .* broadcast_att_zij, dims = 4)
        catty = vcat(catty, reshape(obh, N_head*c_z, N_frames_R,:))
    end
    
    si = l.ipa_linear(catty) 
    return si 
end

"""
Cross IPA Partial Structure Module initialization - single layer - adapted from AF2. From left to right. 
"""
struct IPCrossAStructureModuleLayer
    settings::NamedTuple
    layers::NamedTuple
end
Flux.@functor IPCrossAStructureModuleLayer
function IPCrossAStructureModuleLayer(settings::NamedTuple; dropout_p = 0.1, af = Flux.relu)
    dims = settings.dims
    layers = (
        ipa = IPCrossA(settings),
        ipa_norm = Chain(Dropout(dropout_p), LayerNorm(dims)),
        trans = Chain(Dense(dims => dims, af), Dense(dims => dims, af), Dense(dims => dims)),
        trans_norm = Chain(Dropout(dropout_p), LayerNorm(dims)),
        backbone = BackboneUpdate(dims),
    )
    return IPCrossAStructureModuleLayer(settings, layers)
end

"""
Self IPA Partial Structure Module initialization - single layer - adapted from AF2. 
"""

# We could skip making this struct and just have it be a cross IPA struct. 
struct IPAStructureModuleLayer
    settings::NamedTuple
    layers::NamedTuple
end
Flux.@functor IPAStructureModuleLayer

function IPAStructureModuleLayer(settings::NamedTuple)
    crossL = IPCrossAStructureModuleLayer(settings)
    return IPAStructureModuleLayer(crossL.settings, crossL.layers)
end
function (structuremodulelayer::Union{IPAStructureModuleLayer, IPCrossAStructureModuleLayer} )(T, S; zij = nothing, mask = 0)
    return structuremodulelayer(T, S, T, S; zij = zij, mask = mask)
end

"""
Cross IPA Partial Structure Module - single layer - adapted from AF2. From left to right. 
"""
function (structuremodulelayer::Union{IPCrossAStructureModuleLayer, IPAStructureModuleLayer})(T_L, S_L, T_R, S_R; zij = nothing, mask = 0)
    settings = structuremodulelayer.settings
    if settings.c_z > 0 && zij === nothing
        error("zij must be provided if c_z > 0")
    end

    l = structuremodulelayer.layers
    S_R = S_R .+ l.ipa(T_L, S_L,T_R,S_R, zij = zij, mask = mask) 
    S_R = l.ipa_norm(S_R) 
    S_R = S_R .+ l.trans(S_R) 
    S_R = l.trans_norm(S_R) 
    T_R = l.backbone(T_R, S_R) 
    return T_R, S_R
end

struct IPACache
    sizeL
    sizeR
    batchsize

    # cached arrays
    qh  # channel × head × residues (R) × batch   
    kh  # channel × head × residues (L) × batch
    vh  # channel × head × residues (L) × batch

    qhp  # 3 × {head × query points} × residues (R) × batch
    khp  # 3 × {head × query points} × residues (L) × batch
    vhp  # 3 × {head × point values} × residues (L) × batch
end
Flux.@functor IPACache

"""
    IPACache(settings, batchsize)

Initialize an empty IPA cache.
"""
function IPACache(settings::NamedTuple, batchsize::Integer)
    (; c, N_head, N_query_points, N_point_values) = settings
    qh = zeros(Float32, c, N_head, 0, batchsize)
    kh = zeros(Float32, c, N_head, 0, batchsize)
    vh = zeros(Float32, c, N_head, 0, batchsize)
    qhp = zeros(Float32, 3, N_head * N_query_points, 0, batchsize)
    khp = zeros(Float32, 3, N_head * N_query_points, 0, batchsize)
    vhp = zeros(Float32, 3, N_head * N_point_values, 0, batchsize)
    IPACache(0, 0, batchsize, qh, kh, vh, qhp, khp, vhp)
end

function expand(
    ipa::IPCrossA,
    cache::IPACache,
    TiL::Tuple, siL::AbstractArray, ΔL::Integer,
    TiR::Tuple, siR::AbstractArray, ΔR::Integer;
    zij = nothing,
    mask = 0,
)
    dims, c, N_head, N_query_points, N_point_values, c_z, Typ, pairwise = ipa.settings 
    if haskey(ipa.settings, :use_softmax1) #For compat
        use_softmax1 = ipa.settings.use_softmax1
    else
        use_softmax1 = false
    end
    
    L, R, B = cache.sizeL, cache.sizeR, cache.batchsize
    layer = ipa.layers

    #gamma_h = min.(softplus(layer.gamma_h), 1f2)
    gamma_h = softplus(clamp.(layer.gamma_h,Typ(-100), Typ(100))) #Clamping

    Δqh = reshape(calldense(layer.proj_qh, siR[:,R+1:R+ΔR,:]), (c, N_head, ΔR, B))
    Δkh = reshape(calldense(layer.proj_kh, siL[:,L+1:L+ΔL,:]), (c, N_head, ΔL, B))
    Δvh = reshape(calldense(layer.proj_vh, siL[:,L+1:L+ΔL,:]), (c, N_head, ΔL, B))

    Δqhp = reshape(calldense(layer.proj_qhp, siR[:,R+1:R+ΔR,:]), (3, N_head * N_query_points, ΔR, B))
    Δkhp = reshape(calldense(layer.proj_khp, siL[:,L+1:L+ΔL,:]), (3, N_head * N_query_points, ΔL, B))
    Δvhp = reshape(calldense(layer.proj_vhp, siL[:,L+1:L+ΔL,:]), (3, N_head * N_point_values, ΔL, B))

    kh = cat(cache.kh, Δkh, dims = 3)
    vh = cat(cache.vh, Δvh, dims = 3)

    khp = cat(cache.khp, Δkhp, dims = 3)
    vhp = cat(cache.vhp, Δvhp, dims = 3)

    # calculate inner products
    ΔqhT = permutedims(Δqh, (3, 1, 2, 4))
    kh = permutedims(kh, (1, 3, 2, 4))
    ΔqhTkh = permutedims(batched_mul(ΔqhT, kh), (3, 1, 2, 4))

    # transform vector points to the global frames
    rot_TiL, translate_TiL = TiL
    rot_TiR, translate_TiR = TiR
    ΔTqhp = reshape(T_R3(Δqhp, (rot_TiR[:,:,R+1:R+ΔR,:]), (translate_TiR[:,:,R+1:R+ΔR,:])), (3, N_head, N_query_points, ΔR, B))
    Tkhp = reshape(
        T_R3(reshape(khp, (3, N_head * N_query_points, (L + ΔL) * B)), (rot_TiL[:,:,1:L+ΔL,:]), (translate_TiL[:,:,1:L+ΔL,:])),
        (3, N_head, N_query_points, L + ΔL, B)
    )
    Tvhp = reshape(
        T_R3(reshape(vhp, (3, N_head * N_point_values, (L + ΔL) * B)), (rot_TiL[:,:,1:L+ΔL,:]), (translate_TiL[:,:,1:L+ΔL,:])),
        (3, N_head, N_point_values, L + ΔL, B)
    )

    diffs = unsqueeze(ΔTqhp, dims = 5) .- unsqueeze(Tkhp, dims = 4)
    sum_norms = sumdrop(abs2, diffs, dims = (1, 3))

    w_C = sqrt(2f0 / 9N_query_points)
    dim_scale = sqrt(1f0 / c)
    Δatt_logits = reshape(dim_scale .* ΔqhTkh .- w_C/2 .* gamma_h .* sum_norms, (N_head, ΔR, L + ΔL, B))

    if mask != 0
        mask = unsqueeze((mask[R+1:R+ΔR,1:L+ΔL]), dims = 1)
    end

    if pairwise
        bij = reshape(layer.pair((zij[:,R+1:R+ΔR,1:L+ΔL,:])), (N_head, ΔR, L + ΔL, B))
        w_L = sqrt(1f0/3)
        if use_softmax1
            Δatt = softmax1(w_L .* (Δatt_logits .+ bij) .+ mask, dims = 3)
        else
            Δatt = softmax(w_L .* (Δatt_logits .+ bij) .+ mask, dims = 3)
        end
    else
        w_L = sqrt(1f0/2)
        if use_softmax1
            Δatt = softmax1(w_L .* Δatt_logits .+ mask, dims = 3)
        else
            Δatt = softmax(w_L .* Δatt_logits .+ mask, dims = 3)
        end
    end

    # take the attention weighted sum of the value vectors

    #=
    @show c, N_head, ΔR, L + ΔL, B
    @show size(Δatt)
    @show size(vh)
    @show size(translate_TiR[:,:,R+1:R+ΔR,:])
    @show (1 .- sum(Δatt, dims = 3))
    =#

    oh = sumdrop(
        reshape(Δatt, (1, N_head, ΔR, L + ΔL, B)) .*
        reshape(  vh, (c, N_head,  1, L + ΔL, B)),
        dims = 4,
    )

    if use_softmax1
        ohp_pre = reshape(
            # 3 × N_head × N_point_values × ΔR × batch
                sumdrop(
                    reshape(Δatt,                           (1, N_head, 1,               ΔR, L + ΔL, B)) .*
                    reshape(Tvhp,                           (3, N_head, N_point_values,  1,  L + ΔL, B)),
                    dims =5,
                )  .+
                    reshape(translate_TiR[:,:,R+1:R+ΔR,:],  (3, 1,      1,               ΔR, B)) .*
                    reshape(1 .- sum(Δatt, dims = 3),       (1, N_head, 1,               ΔR, B)),
            (3, N_head * N_point_values, ΔR * B)
        )
    else
        ohp_pre = reshape(
            # 3 × N_head × N_point_values × ΔR × batch
            sumdrop(
                reshape(Δatt, (1, N_head,              1, ΔR, L + ΔL, B)) .*
                reshape(Tvhp, (3, N_head, N_point_values,  1, L + ΔL, B)),
                dims = 5,
            ),
            (3, N_head * N_point_values, ΔR * B)
        )
    end

    ohp = reshape(
        T_R3_inv(
            ohp_pre
            ,
            (rot_TiR[:,:,R+1:R+ΔR,:]),
            (translate_TiR[:,:,R+1:R+ΔR,:])
        ),
        (3, N_head, N_point_values, ΔR, B)
    )
    ohp_norms = sqrt.(sumdrop(abs2, ohp, dims = 1) .+ Typ(0.000001f0))
    
    # concatenate all outputs
    o = [
        reshape(oh, (c * N_head, ΔR, B))
        reshape(ohp, (3 * N_head * N_point_values, ΔR, B))
        reshape(ohp_norms, (N_head * N_point_values, ΔR, B))
    ]
    if pairwise
        o = [
            o
            reshape(
                sumdrop(
                    reshape(                           Δatt, (  1, N_head, ΔR, L + ΔL, B)) .*
                    reshape((zij[:,R+1:R+ΔR,1:L+ΔL,:]), (c_z,      1, ΔR, L + ΔL, B)),
                    dims = 4
                ),
                (c_z * N_head, ΔR, B)
            )
        ]
    end

    cache = IPACache(
        L + ΔL,
        R + ΔR,
        B,
        cat(cache.qh, Δqh, dims = 3),
        cat(cache.kh, Δkh, dims = 3),
        cat(cache.vh, Δvh, dims = 3),
        cat(cache.qhp, Δqhp, dims = 3),
        cat(cache.khp, Δkhp, dims = 3),
        cat(cache.vhp, Δvhp, dims = 3),
    )
    layer.ipa_linear(o), cache
end

sumdrop(x; dims) = dropdims(sum(x; dims); dims)
sumdrop(f, x; dims) = dropdims(sum(f, x; dims); dims)

# dense(x) to avoid https://github.com/FluxML/Flux.jl/issues/2407
function calldense(dense::Dense, x::AbstractArray)
    d1 = size(dense.weight, 1)
    reshape(dense(reshape(x, size(x, 1), :)), d1, size(x)[2:end]...)
end
