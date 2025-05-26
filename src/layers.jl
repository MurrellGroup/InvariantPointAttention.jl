"""
Projects the frame embedding => 6, and uses this to transform the input frames.
"""
struct BackboneUpdate
    layers::NamedTuple
end

Flux.@layer BackboneUpdate

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
    IPA_settings(
        dims;
        c = 16,
        N_head = 12,
        N_query_points = 4,
        N_point_values = 8,
        c_z = 0,
        Typ = Float32,
        use_softmax1 = false,
        scaling_qk = :default,
    )

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
    use_softmax1 = false,
    scaling_qk = :default, # :none, :default, or a vector of length N_head\
) = (;
    dims,
    c,
    N_head,
    N_query_points,
    N_point_values,
    c_z,
    Typ,
    pairwise = c_z > 0,
    use_softmax1,
    scaling_qk,
)


"""
    IPCrossA(settings)

Invariant Point Cross Attention (IPCrossA). Information flows from L (Keys, Values) to R (Queries).

Get `settings` with [`IPA_settings`](@ref)
"""
struct IPCrossA
    settings::NamedTuple
    layers::NamedTuple
end

Flux.@layer IPCrossA # provides parameter collection, gpu movement and more

function IPCrossA(settings::NamedTuple; rope_kwargs...)
    dims, c, N_head, N_query_points, N_point_values, c_z, Typ, pairwise = settings 
    # Needs a slightly unusual initialization - hat-tip: Kenta
    init = Flux.kaiming_uniform(gain = 1.0f0)
    if pairwise
        pair = Dense(c_z => N_head, bias = false; init)
        ipa_linear = Dense(N_head*c_z + N_head*c + 4*N_head*N_point_values => dims)
    else
        pair = nothing
        ipa_linear = Dense(N_head*c + 4*N_head*N_point_values => dims)
    end
    scale_h = if settings.scaling_qk == :none
        nothing
    else
        v = if settings.scaling_qk == :default
            0.1 .+ range(0, 1, N_head).^2
        elseif settings.scaling_qk isa AbstractVector{<:Real}
            settings.scaling_qk
        end
        Typ.(repeat(v, outer = N_query_points))
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
        gamma_h = min.(ones(Typ, N_head) .* Typ(0.541), 1f2),
        scale_h = scale_h,
    )
    return IPCrossA(settings, layers)
end


# We could skip making this struct and just have it be a cross IPA struct. 
"""
Strictly Self-IPA initialization
"""
struct IPA
    settings::NamedTuple
    layers::NamedTuple
end

Flux.@layer IPA

function IPA(settings::NamedTuple)
    crossL = IPCrossA(settings)
    return IPA(crossL.settings, crossL.layers)
end

function (ipa::Union{IPA, IPCrossA})(T::Tuple{AbstractArray,AbstractArray}, S::AbstractArray; Z = nothing, mask = 0)
    return ipa(T, S, T, S; zij = Z, mask = mask)
end

#Attention props from L (Keys, Values) to R (Queries).
#Because IPA uses Q'K, our pairwise matrices are R-by-L 
#Rope is an IPARoPE, applying the usual RoPE to queries and keys pertaining to the same chains and a fixed rotation to queries and keys pertaining to different chains. 
#Chain diffs defaults to 1, meaning everything is in the same chain. Otherwise, a pairwise matrix where 1 denotes the same chain, 0 denotes different chains should be used. 
function (ipa::Union{IPCrossA, IPA})(
    TiL::Tuple{AbstractArray, AbstractArray}, siL::AbstractArray,
    TiR::Tuple{AbstractArray, AbstractArray}, siR::AbstractArray;
    zij = nothing, mask = 0, customgrad = true, 
    rope::Union{IPARoPE, Nothing} = nothing, chain_diffs = 1, show_warnings = false, old_eucdists = false
)
    if mask == 0 || siL != siR || TiL != TiR
        if show_warnings
            @warn "Forcing customgrad to false"
        end
        customgrad = false 
    end

    if customgrad  && old_eucdists
        return ipa_customgrad(ipa, TiL, siL, zij, mask, rope = rope, chain_diffs = chain_diffs)
    end

    if !isnothing(zij)
        #This is assuming the dims of zij are c, N_frames_L, N_frames_R, batch
        size(zij,2) == size(siR,2) || throw(DimensionMismatch("zij and siR size mismatch"))
        size(zij,3) == size(siL,2) || throw(DimensionMismatch("zij and siL size mismatch")) 
    end
    if mask != 0
        size(mask,1) == size(siR, 2) || throw(DimensionMismatch("mask and siR size mismatch"))
        size(mask,2) == size(siL, 2) || throw(DimensionMismatch("mask and siL size mismatch"))
    end
    
    batch = size(siL, 3) 

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
    if !isnothing(rope)
        qhTkh = dotproducts(rope, qh, kh; chain_diffs)
    else
        qhTkh = dotproducts(qh, kh)
    end 

    vh = reshape(l.proj_vh(siL),(c,N_head,N_frames_L,:))

    if isnothing(l.scale_h)
        qhp = reshape(l.proj_qhp(siR),(3,N_head*N_query_points,N_frames_R,:))
        khp = reshape(l.proj_khp(siL),(3,N_head*N_query_points,N_frames_L,:))
    else
        scale_h = reshape(l.scale_h, (1,N_head*N_query_points,1,1))
        qhp = reshape(l.proj_qhp(siR),(3,N_head*N_query_points,N_frames_R,:)) .* scale_h
        khp = reshape(l.proj_khp(siL),(3,N_head*N_query_points,N_frames_L,:)) .* scale_h
    end
    vhp = reshape(l.proj_vhp(siL),(3,N_head*N_point_values,N_frames_L,:))

    # This should be Q'K, following IPA, which isn't like the regular QK'
    # Dot products between queries and keys.
                        #FramesR, c, N_head, Batch

    

    # Applying our transformations to the queries, keys, and values to put them in the global frame.
    Tqhp = reshape(T_R3(qhp, rot_TiR,translate_TiR),3,N_head,N_query_points,N_frames_R,:) 
    Tkhp = reshape(T_R3(khp, rot_TiL,translate_TiL),3,N_head,N_query_points,N_frames_L,:)
    Tvhp = T_R3(vhp, rot_TiL, translate_TiL)

    if old_eucdists # BREAKING: THIS PATH IS USED IN RUNTESTS
        diffs_glob = Flux.unsqueeze(Tqhp, dims = 5) .- Flux.unsqueeze(Tkhp, dims = 4)
        sum_norms_glob = reshape(sum(abs2, diffs_glob, dims = [1,3]),N_head,N_frames_R,N_frames_L,:) #Sum over points for each head
    else 
        PTqhp = permutedims(Tqhp, (4,1,3,2,5)) # NR, 3, Nqp, Nh, batch_size 
        PTkhp = permutedims(Tkhp, (1,3,4,2,5)) # 3, Nqp, NL, Nh, batch_size
        RPTqhp = reshape(PTqhp, N_frames_R, 3*N_query_points, N_head*batch)
        RPTkhp = reshape(PTkhp, 3*N_query_points, N_frames_L, N_head*batch) 
        sum_norms_glob = permutedims(reshape(PairwiseEuclideans(RPTqhp, RPTkhp), N_frames_R, N_frames_L, N_head, batch), (3,1,2,4)) 
    end


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
        mask = Flux.unsqueeze(mask, dims = 1) 
    end

    if use_softmax1
        att = softmax1(w_L .* (att_arg .+ bij) .+ mask, dims = 3)
    else
        att = Flux.softmax(w_L .* (att_arg .+ bij) .+ mask, dims = 3)
    end
    # Applying the attention weights to the values.
    oh = permutedims(batched_mul(permutedims(att,(2,3,1,4)), permutedims(vh,(3,1,2,4))),(2,3,1,4));
    
    broadcast_att_ohp = reshape(att,(1,N_head,1,N_frames_R,N_frames_L,:))
    broadcast_tvhp = reshape(Tvhp,(3,N_head,N_point_values,1,N_frames_L,:))

    if use_softmax1
        pre_ohp_r = sum(broadcast_att_ohp.*broadcast_tvhp,dims=5)
        unreshaped_ohp_r = pre_ohp_r .+ (1 .- sum(broadcast_att_ohp, dims = 5)) .* reshape(translate_TiR, 3, 1, 1, N_frames_R, 1, :)
        ohp_r = reshape(unreshaped_ohp_r,3,N_head*N_point_values,N_frames_R,:)
    else
        ohp_r = reshape(sum(broadcast_att_ohp.*broadcast_tvhp,dims=5),3,N_head*N_point_values,N_frames_R,:)
    end

    #ohp_r were in the global frame, so we put those ba ck in the recipient local
    ohp = T_R3_inv(ohp_r, rot_TiR, translate_TiR) 
    normed_ohp = sqrt.(sum(abs2, ohp,dims = 1) .+ Typ(0.000001f0)) #Adding eps

    catty = vcat(
        reshape(oh, N_head*c, N_frames_R,:),
        reshape(ohp, 3*N_head*N_point_values, N_frames_R,:),
        reshape(normed_ohp, N_head*N_point_values, N_frames_R,:)
        ) 

    if pairwise
        obh = batched_mul(permutedims(zij,(1,3,2,4)), permutedims(att,(3,1,2,4)))
        catty = vcat(catty, reshape(obh, N_head*c_z, N_frames_R,:))
    end

    si = l.ipa_linear(catty) 
    return si 
end

function ipa_customgrad(
    ipa::Union{IPCrossA, IPA}, 
    Ti::Tuple{AbstractArray,AbstractArray}, 
    S::AbstractArray, 
    zij::Union{AbstractArray, Nothing},
    mask::AbstractArray; 
    rope = nothing, 
    chain_diffs = 1)         
    # Get relevant parameters from our ipa struct.
    l = ipa.layers
    dims, c, N_head, N_query_points, N_point_values, c_z, Typ, pairwise = ipa.settings 
    if haskey(ipa.settings, :use_softmax1) #For compat
        use_softmax1 = ipa.settings.use_softmax1
    else
        use_softmax1 = false
    end

    TiL = Ti
    TiR = Ti
    if size(Ti[2],2) != 1
        TiL = TiR = (Ti[1], reshape(Ti[2], size(Ti[2],1), 1, size(Ti[2])[2:end]...)) 
    end
    siL = S
    siR = S
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

    if isnothing(l.scale_h)
        qhp = reshape(l.proj_qhp(siR),(3,N_head*N_query_points,N_frames_R,:))
        khp = reshape(l.proj_khp(siL),(3,N_head*N_query_points,N_frames_L,:))
    else
        scale_h = reshape(l.scale_h, (1,N_head*N_query_points,1,1))
        qhp = reshape(l.proj_qhp(siR),(3,N_head*N_query_points,N_frames_R,:)) .* scale_h
        khp = reshape(l.proj_khp(siL),(3,N_head*N_query_points,N_frames_L,:)) .* scale_h
    end

    vhp = reshape(l.proj_vhp(siL),(3,N_head*N_point_values,N_frames_L,:))
    
    if !isnothing(rope)
       # @show size(qh)
       #@show size(kh)
        qhTkh = dotproducts(rope, qh, kh; chain_diffs)
    else
        qhTkh = dotproducts(qh, kh)
    end 

    # Applying our transformations to the queries, keys, and values to put them in the global frame.
    Tqhp = reshape(T_R3(qhp, rot_TiR,translate_TiR),3,N_head,N_query_points,N_frames_R,:) 
    Tkhp = reshape(T_R3(khp, rot_TiL,translate_TiL),3,N_head,N_query_points,N_frames_L,:)
    Tvhp = T_R3(vhp, rot_TiL, translate_TiL)

    diffs_glob = pair_diff(Tqhp, Tkhp, dims = 4)
    sum_norms_glob = reshape(sumabs2(diffs_glob, dims = [1,3]),N_head,N_frames_R,N_frames_L,:) #Sum over points for each head
    

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
        mask = Flux.unsqueeze(mask, dims = 1) 
    end

    if use_softmax1
        att = softmax1(w_L .* (att_arg .+ bij) .+ mask, dims = 3)
    else
        att = Flux.softmax(w_L .* (att_arg .+ bij) .+ mask, dims = 3)
    end

    # can save one allocation here with a grad
    oh = permutedims(batched_mul(permutedims(att,(2,3,1,4)), permutedims(vh,(3,1,2,4))),(2,3,1,4));
    # This part needs a GPU kernel or a particularly clever matmul, without it the grad costs 7 gpu allocations
    broadcast_att_ohp = reshape(att,(1,N_head,1,N_frames_R,N_frames_L,:))
    broadcast_tvhp = reshape(Tvhp,(3,N_head,N_point_values,1,N_frames_L,:))
    
    if use_softmax1
        pre_ohp_r = sum(broadcast_att_ohp.*broadcast_tvhp,dims=5)
        # customgrad for this wouldn't save much 
        unreshaped_ohp_r = pre_ohp_r .+ (1 .- sum(broadcast_att_ohp, dims = 5)) .* reshape(translate_TiR, 3, 1, 1, N_frames_R, 1, :)
        ohp_r = reshape(unreshaped_ohp_r,3,N_head*N_point_values,N_frames_R,:)
    else
        ohp_r = reshape(sum(broadcast_att_ohp.*broadcast_tvhp,dims=5),3,N_head*N_point_values,N_frames_R,:)
    end
    #ohp_r were in the global frame, so we put those back in the recipient local
    ohp = T_R3_inv(ohp_r, rot_TiR, translate_TiR) 
    normed_ohp = sqrt.(sumabs2(ohp, dims = 1) .+ Typ(0.000001f0)) #Adding eps
    catty = vcat(
        reshape(oh, N_head*c, N_frames_R,:),
        reshape(ohp, 3*N_head*N_point_values, N_frames_R,:),
        reshape(normed_ohp, N_head*N_point_values, N_frames_R,:)
        ) 
    if pairwise
        obh = batched_mul(permutedims(zij,(1,3,2,4)), permutedims(att,(3,1,2,4)))
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
Flux.@layer IPCrossAStructureModuleLayer
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

# We could skip making this struct and just have it be a cross IPA struct. 
"""
Self IPA Partial Structure Module initialization - single layer - adapted from AF2. 
"""
struct IPAStructureModuleLayer
    settings::NamedTuple
    layers::NamedTuple
end
Flux.@layer IPAStructureModuleLayer

function IPAStructureModuleLayer(settings::NamedTuple)
    crossL = IPCrossAStructureModuleLayer(settings)
    return IPAStructureModuleLayer(crossL.settings, crossL.layers)
end
function (structuremodulelayer::Union{IPAStructureModuleLayer, IPCrossAStructureModuleLayer} )(T, S; zij = nothing, mask = 0)
    return structuremodulelayer(T, S, T, S; zij = zij, mask = mask)
end

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
Flux.@layer IPACache

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
    rope = nothing
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
    if !isnothing(rope)
        Δqh = rope(Δqh)
        Δkh = rope(Δkh)
    end
    Δvh = reshape(calldense(layer.proj_vh, siL[:,L+1:L+ΔL,:]), (c, N_head, ΔL, B))

    Δqhp = reshape(calldense(layer.proj_qhp, siR[:,R+1:R+ΔR,:]), (3, N_head * N_query_points, ΔR, B))
    Δkhp = reshape(calldense(layer.proj_khp, siL[:,L+1:L+ΔL,:]), (3, N_head * N_query_points, ΔL, B))

    if isnothing(layer.scale_h)
        Δqhp = reshape(calldense(layer.proj_qhp, siR[:,R+1:R+ΔR,:]), (3, N_head * N_query_points, ΔR, B))
        Δkhp = reshape(calldense(layer.proj_khp, siL[:,L+1:L+ΔL,:]), (3, N_head * N_query_points, ΔL, B))
    else
        scale_h = reshape(layer.scale_h, (1,N_head*N_query_points,1,1))
        Δqhp = reshape(calldense(layer.proj_qhp, siR[:,R+1:R+ΔR,:]), (3, N_head * N_query_points, ΔR, B)) .* scale_h
        Δkhp = reshape(calldense(layer.proj_khp, siL[:,L+1:L+ΔL,:]), (3, N_head * N_query_points, ΔL, B)) .* scale_h
    end

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
        T_R3(reshape(khp, (3, N_head * N_query_points, (L + ΔL), B)), (rot_TiL[:,:,1:L+ΔL,:]), (translate_TiL[:,:,1:L+ΔL,:])),
        (3, N_head, N_query_points, L + ΔL, B)
    )
    Tvhp = reshape(
        T_R3(reshape(vhp, (3, N_head * N_point_values, (L + ΔL), B)), (rot_TiL[:,:,1:L+ΔL,:]), (translate_TiL[:,:,1:L+ΔL,:])),
        (3, N_head, N_point_values, L + ΔL, B)
    )

    diffs = Flux.unsqueeze(ΔTqhp, dims = 5) .- Flux.unsqueeze(Tkhp, dims = 4)
    sum_norms = sumdrop(abs2, diffs, dims = (1, 3))

    w_C = sqrt(2f0 / 9N_query_points)
    dim_scale = sqrt(1f0 / c)
    Δatt_logits = reshape(dim_scale .* ΔqhTkh .- w_C/2 .* gamma_h .* sum_norms, (N_head, ΔR, L + ΔL, B))

    if mask != 0
        mask = Flux.unsqueeze((mask[R+1:R+ΔR,1:L+ΔL]), dims = 1)
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
            (3, N_head * N_point_values, ΔR, B)
        )
    else
        ohp_pre = reshape(
            # 3 × N_head × N_point_values × ΔR × batch
            sumdrop(
                reshape(Δatt, (1, N_head,              1, ΔR, L + ΔL, B)) .*
                reshape(Tvhp, (3, N_head, N_point_values,  1, L + ΔL, B)),
                dims = 5,
            ),
            (3, N_head * N_point_values, ΔR, B)
        )
    end

    ohp = reshape(
        T_R3_inv(
            ohp_pre,
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
