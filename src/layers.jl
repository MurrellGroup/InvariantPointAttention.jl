# TODO: Q/K point scaling

#=
TODO: RoPE
Rope is an IPARoPE, applying the usual RoPE to queries and keys pertaining to the
same chains and a fixed rotation to queries and keys pertaining to different chains. 
Chain diffs defaults to 1, meaning everything is in the same chain. Otherwise, a
pairwise matrix where 1 denotes the same chain, 0 denotes different chains should be used. 
=#

# Algorithm 23 Backbone update
struct BackboneUpdate{T<:Dense}
    linear::T
end

BackboneUpdate(s_dim::Int) = BackboneUpdate(Dense(s_dim => 6))

(bu::BackboneUpdate)(frames::Rigid, s) =
    update_frame(frames, reshape(bu.linear(s), 3, 2, size(s, 2), :))


Base.@kwdef struct IPAConfig
    n_dims_s
    n_dims_z
    c = 16
    N_head = 12
    N_query_points = 4
    N_point_values = 8
    softmax = Flux.softmax
end

struct IPA
    config::IPAConfig
    wqh::Dense
    wkh::Dense
    wvh::Dense
    wqhp::Dense
    wkhp::Dense
    wvhp::Dense
    pair::Dense
    ipa_linear::Dense
    gamma_h::Vector{Float32}
end

function IPA(config::IPAConfig; rope_kwargs...)
    @unpack n_dims_s, c, N_head, N_query_points, N_point_values, n_dims_z, = config
    init = Flux.kaiming_uniform(gain = 1.0f0)
    pair = Dense(n_dims_z => N_head, bias = false; init)
    ipa_linear = Dense(N_head * (c + n_dims_z) + 4 * N_head * N_point_values => n_dims_s)
    return IPA(config,
        Dense(n_dims_s => c * N_head; bias=false, init),
        Dense(n_dims_s => c * N_head; bias=false, init),
        Dense(n_dims_s => c * N_head; bias=false, init),
        Dense(n_dims_s => 3 * N_head * N_query_points; bias=false, init),
        Dense(n_dims_s => 3 * N_head * N_query_points; bias=false, init),
        Dense(n_dims_s => 3 * N_head * N_point_values; bias=false, init),
        pair,
        ipa_linear,
        min.(ones(Float32, N_head) .* 0.541f0, 100f0),
    )
end

function (ipa::IPA)(
    framesQ::Rigid, sQ::AbstractArray{T},
    framesK::Rigid, sK::AbstractArray{T},
    z=nothing; mask=nothing
) where T
    @unpack n_dims_z, c, N_head, N_query_points, N_point_values = ipa.config
    @unpack wqh, wkh, wvh, wqhp, wkhp, wvhp, pair, ipa_linear, gamma_h = ipa

    do_pairwise = !isnothing(z)

    N_q = size(sQ, 2)
    N_k = size(sK, 2)

    gamma_h = softplus(clamp.(gamma_h, -100, 100))
    w_C     = √(2 / (9 * N_query_points)) |> T

    qh = reshape(wqh(sQ), c, N_head, N_q, :)
    kh = reshape(wkh(sK), c, N_head, N_k, :)
    vh = reshape(wvh(sK), c, N_head, N_k, :)

    qhTkh = permutedims(
        batched_mul(
            permutedims(qh, (3, 1, 2, 4)),  # (N_q, c, N_head, ...)
            permutedims(kh, (1, 3, 2, 4))   # (c, N_k, N_head, ...)
        ),
        (3, 1, 2, 4)                        # (N_head, N_q, N_k, ...)
    )

    qhp_local = reshape(wqhp(sQ), 3, N_head*N_query_points, N_q, :)
    khp_local = reshape(wkhp(sK), 3, N_head*N_query_points, N_k, :)
    vhp_local = reshape(wvhp(sK), 3, N_head*N_point_values, N_k, :)

    qhp = reshape(framesQ * qhp_local, 3, N_head, N_query_points, N_q, 1, :)
    khp = reshape(framesK * khp_local, 3, N_head, N_query_points, 1, N_k, :)
    vhp = framesK * vhp_local

    distances = reshape(
        sum(abs2, qhp .- khp, dims=(1,3)),
        N_head, N_q, N_k, :
    )

    if do_pairwise
        w_L = sqrt(1/3) |> T
        b   = reshape(pair(z), N_head, N_q, N_k, :)
    else
        w_L = sqrt(1/2) |> T
        b   = false
    end

    # (N_head, N_q, N_k, ...)
    att_logits = T(1/√(c)) .* qhTkh .- w_C/2 .* (gamma_h .* distances)
    if b !== false
        att_logits .= att_logits .+ b
    end

    att_logits = reshape(att_logits, N_head, N_q, N_k, :)
    if mask !== nothing
        att_logits .+= Flux.unsqueeze(mask, dims=1)
    end

    att = ipa.config.softmax(w_L .* att_logits, dims=3) # softmax over K dimension

    oh = permutedims(
        batched_mul(
            permutedims(att, (2, 3, 1, 4)),  # (N_q, N_k, N_head, ...)
            permutedims(vh,  (3, 1, 2, 4))   # (N_k, c, N_head, ...)
        ),
        (2, 3, 1, 4)                         # (c, N_head, N_q, ...)
    )

    # att => (N_head, N_q, N_k, ...)
    # we reshape it to broadcast over vhp => (3, N_head, N_point_values, N_k, ...)
    att_broadcast        = reshape(att, 1, N_head, 1, N_q, N_k, :)
    vhp_global_broadcast = reshape(vhp, 3, N_head, N_point_values, 1, N_k, :)

    if ipa.config.softmax isa typeof(softmax1)
        pre_ohp_global = sum(att_broadcast .* vhp_global_broadcast, dims=5)  # sum over K
        leftover_frac  = 1 .- sum(att_broadcast, dims=5)                     # shape => (1, N_head, 1, N_q, 1, ...)
        offset = leftover_frac .* reshape(values(translation(framesQ)), 3, 1, 1, N_q, 1, :)
        ohp_global = pre_ohp_global .+ offset
        ohp_r = reshape(ohp_global, 3, N_head*N_point_values, N_q, :)
    else
        ohp_r = reshape(
            sum(att_broadcast .* vhp_global_broadcast, dims=5),
            3, N_head*N_point_values, N_q, :
        )
    end

    ohp_local = inverse(framesQ) * ohp_r

    normed_ohp = .√(sum(abs2, ohp_local, dims=1) .+ 1f-6)

    # Flatten oh => (c*N_head, N_q, ...)
    oh_reshaped       = reshape(oh, c*N_head, N_q, :)
    # Flatten ohp => (3*N_head*N_point_values, N_q, ...)
    ohp_reshaped      = reshape(ohp_local, 3*N_head*N_point_values, N_q, :)
    # Flatten normed_ohp => (N_head*N_point_values, N_q, ...)
    ohp_norm_reshaped = reshape(normed_ohp, N_head*N_point_values, N_q, :)

    # Gather partial outputs
    outputs = [
        oh_reshaped,
        ohp_reshaped,
        ohp_norm_reshaped
    ]

    if do_pairwise
        obh = batched_mul(
            permutedims(z,   (1, 3, 2, 4)),  # => (n_dims_z, N_k, N_q, ...)
            permutedims(att, (3, 1, 2, 4)))  # => (N_k, N_head, N_q, ...)
        push!(outputs, reshape(obh, N_head*n_dims_z, N_q, :))
    end

    catty = vcat(outputs...)
    s     = ipa_linear(catty)

    return s
end

(ipa::IPA)(frames::Rigid, s::AbstractArray, args...; kwargs...) =
    ipa(frames, s, frames, s, args...; kwargs...)

struct StructureModule
    ipa::IPA
    ipa_norm::Chain
    trans::Chain
    trans_norm::Chain
    backbone::BackboneUpdate
end

function StructureModule(config::IPAConfig; dropout_p=0.1, af=Flux.relu)
    dims = config.n_dims_s
    return StructureModule(
        IPA(config),
        Chain(Dropout(dropout_p), LayerNorm(dims)),
        Chain(Dense(dims => dims, af), Dense(dims => dims, af), Dense(dims => dims)),
        Chain(Dropout(dropout_p), LayerNorm(dims)),
        BackboneUpdate(dims))
end

function (structuremodule::StructureModule)(
    framesQ::Rigid, sQ::AbstractArray{T},
    framesK::Rigid, sK::AbstractArray{T},
    z=nothing; mask=nothing
) where T
    @unpack ipa, ipa_norm, trans, trans_norm, backbone = structuremodule
    sQ = sQ .+ ipa(framesQ, sQ, framesK, sK, z; mask)
    sQ = ipa_norm(sQ)
    sQ = sQ .+ trans(sQ)
    sQ = trans_norm(sQ)
    return backbone(framesQ, sQ), sQ
end

(structuremodule::StructureModule)(framesQ::Rigid, sQ::AbstractArray, args...; kwargs...) =
    structuremodule(framesQ, sQ, framesQ, sQ, args...; kwargs...)
