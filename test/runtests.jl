using InvariantPointAttention
using InvariantPointAttention: get_rotation, get_translation
using InvariantPointAttention: softmax1
using Zygote:gradient
using Zygote:withgradient
using Flux: params
using Flux: batched_mul
using InvariantPointAttention: T_R3, T_R3_inv, _T_R3_no_rrule, _T_R3_inv_no_rrule, diff_sum_glob, _diff_sum_glob_no_rrule, pair_diff, _pair_diff_no_rrule
using InvariantPointAttention: L2norm, _L2norm_no_rrule, sumabs2, _sumabs2_no_rrule
using Test

@testset "InvariantPointAttention.jl" begin
    # Write your tests here.
    @testset "Softmax1" begin
        #Check if softmax1 is consistent with softmax, when adding an additional zero logit
        x = randn(4,3)
        xaug = hcat(x, zeros(4,1))
        @test InvariantPointAttention.softmax1(x, dims = 2) ≈ InvariantPointAttention.Flux.softmax(xaug, dims = 2)[:,1:end-1]
    end
    @testset "Softmax1 custom grad" begin
        x = randn(3,10,41,13)
 
        function softmax1_no_rrule(x::AbstractArray{T}; dims = 1) where {T}
            _zero = T(0)
            max_ = max.(maximum(x; dims), _zero)
            @fastmath out = exp.(x .- max_)
            tmp = sum(out, dims = dims)
            out ./ (tmp + exp.(-max_))
        end

        for k in 1:4
            f(x; dims = k) = sum(softmax1(x; dims))
            g(x; dims = k) = sum(softmax1_no_rrule(x; dims))
            @test gradient(f, x)[1] ≈ gradient(g, x)[1]
        end
    end  
    @testset "text o-comps" begin
        N = 10 
        batch_size = 3
        N_h = 8 
        c_z = 5
        c = 3
        sumdrop(x; dims) = dropdims(sum(x; dims); dims)
        N_head = N_h
        N_frames_R = N_frames_L = N
        
        vh = randn(c, N_h, N, batch_size)
        zij = randn(c_z, N, N, batch_size)
        att = randn(N_h, N, N, batch_size)
        
        obh = batched_mul(permutedims(zij,(1,3,2,4)), permutedims(att,(3,1,2,4)))
        oh = permutedims(batched_mul(permutedims(att,(2,3,1,4)), permutedims(vh,(3,1,2,4))),(2,3,1,4))
        
        broadcast_att_oh = reshape(att,(1,N_head,N_frames_R,N_frames_L,:))
        broadcast_vh = reshape(vh, (c,N_head,1,N_frames_L,:))
        oh2 = reshape(sumdrop(broadcast_att_oh .* broadcast_vh,dims = 4), c,N_head,N_frames_R,:)
        
        broadcast_zij = reshape(zij,(c_z,1,N_frames_R,N_frames_L,:))
        broadcast_att_zij = reshape(att,(1,N_head,N_frames_R,N_frames_L,:))
        obh2 = sumdrop(broadcast_zij .* broadcast_att_zij, dims = 4)
        
        @test obh .- obh2 < 1e-12
        @test oh .- oh2 < 1e-12
    end
    @testset "T_R3 custom grad" begin 
        x = randn(3,5,10,15)
        rot = get_rotation(10,15) 
        trans = get_translation(10,15)

        @test gradient(sum ∘ T_R3, x, rot, trans)[1] ≈ gradient(sum ∘ _T_R3_no_rrule, x, rot, trans)[1]
    end

    @testset "T_R3_inv custom grad" begin 
        x = randn(3,5,10,15)
        rot = get_rotation(10,15) 
        trans = get_translation(10,15)
        @test gradient(sum ∘ T_R3_inv, x, rot, trans)[1] ≈ gradient(sum ∘ _T_R3_inv_no_rrule, x, rot, trans)[1]
    end

    @testset "sumabs2 custom grad" begin 
        x = randn(3,10,41,13)

        for k in 1:4
            f(x; dims = k) = sum(sumabs2(x; dims))
            g(x; dims = k) = sum(sum(abs2, x; dims))
            cval, cgs = withgradient(f, x)
            val, gs = withgradient(g, x)
            @test cval ≈ val
            @test keys(cgs) ≈ keys(gs)
        end
    end

    @testset "L2norm custom grad" begin 
        x = randn(3,10,41,13)

        for k in 1:4
            f(x; dims = k) = sum(L2norm(x; dims, eps = 0.1f0))
            g(x; dims = k) = sum(_L2norm_no_rrule(x; dims, eps = 0.1f0))
            cval, cgs = withgradient(f, x)
            val, gs = withgradient(g, x)
            @test cval ≈ val
            @test keys(cgs)[1] ≈ keys(gs)[1]
        end
    end

    @testset "pair_diff custom grad" begin 
        x = randn(3,5,5,5,5)
        y = randn(3,5,5,15,5)
        @test gradient(sum ∘ pair_diff, x, y)[1] ≈ gradient(sum ∘ _pair_diff_no_rrule, x, y)[1]
    end

    @testset "ipa_customgrad" begin
        batch_size = 3
        framesL = 10
        framesR = 10
        dim = 10
        
        siL = Float64.(randn(dim,framesL,batch_size)) 
        siR = siL
        # Use CLOPS.jl shape notation
        TiL = (Float64.(get_rotation(framesL,batch_size)), randn(Float64, 3,framesL,batch_size)) 
        TiR = TiL 
        zij = randn(Float64, 16, framesR, framesL, batch_size) 

        ipa = IPCrossA(IPA_settings(dim; use_softmax1 = true, c_z = 16, Typ = Float64))  
        # Batching on mask
        mask = right_to_left_mask(framesL)[:,:,repeat(1:1, inner = batch_size)]
        ps = params(ipa)
        
        lz,gs = withgradient(ps) do 
            sum(ipa(TiL, siL, TiR, siR; zij, mask, customgrad = true))
        end
        
        lz2, zygotegs = withgradient(ps) do 
            sum(ipa(TiL, siL, TiR, siR; zij, mask, customgrad = false))
        end
        
        for (gs, zygotegs) in zip(keys(gs),keys(zygotegs))
            @test maximum(abs.(gs .- zygotegs)) < 1f-5
        end
        #@show lz, lz2
        @test abs.(lz - lz2) < 1f-5
    end
    @testset "IPAsoftmax_invariance" begin
        batch_size = 3
        framesL = 100
        framesR = 101
        dim = 768
        
        siL = Float32.(randn(dim,framesL,batch_size)) 
        siR = Float32.(randn(dim,framesR,batch_size))
        
        T_locL = (get_rotation(framesL,batch_size), get_translation(framesL,batch_size)) 
        T_locR = (get_rotation(framesR,batch_size), get_translation(framesR,batch_size)) 

        # Get 1 global SE(3) transformation for each batch.
        T_glob = (get_rotation(batch_size), get_translation(batch_size))
        T_GlobL = (stack([T_glob[1] for i in 1:framesL],dims = 3), stack([T_glob[2] for i in 1:framesL],dims = 3))
        T_GlobR = (stack([T_glob[1] for i in 1:framesR],dims = 3), stack([T_glob[2] for i in 1:framesR],dims = 3))
        
        T_newL = InvariantPointAttention.T_T(T_GlobL,T_locL)
        T_newR = InvariantPointAttention.T_T(T_GlobR,T_locR)
        
        ipa = IPAStructureModuleLayer(IPA_settings(dim; use_softmax1 = true)) 
        
        T_loc, si_loc = ipa(T_locL,siL, T_locR, siR)
        T_glob, si_glob = ipa(T_newL, siL, T_newR, siR)
        @test si_glob ≈ si_loc
    end

    @testset "IPACache" begin
        dims = 8
        c_z = 2
        settings = IPA_settings(dims; c_z)
        ipa = IPCrossA(settings)

        # generate random data
        L = 5
        R = 6
        B = 4
        siL = randn(Float32, dims, L, B)
        siR = randn(Float32, dims, R, B)
        zij = randn(Float32, c_z, R, L, B)
        TiL = (get_rotation(L, B), get_translation(L, B))
        TiR = (get_rotation(R, B), get_translation(R, B))

        # check the consistency
        cache = InvariantPointAttention.IPACache(settings, B)
        siR′, cache′ = InvariantPointAttention.expand(ipa, cache, TiL, siL, L, TiR, siR, R; zij)
        @test size(siR′) == size(siR)
        @test siR′ ≈ ipa(TiL, siL, TiR, siR; zij)

        # calculate in two steps
        cache = InvariantPointAttention.IPACache(settings, B)
        siR1, cache = InvariantPointAttention.expand(ipa, cache, TiL, siL, L, TiR, siR, 2; zij)
        siR2, cache = InvariantPointAttention.expand(ipa, cache, TiL, siL, 0, TiR, siR, 4; zij)
        @test cat(siR1, siR2, dims = 2) ≈ ipa(TiL, siL, TiR, siR; zij)
    end

    @testset "IPACache_v2" begin
        dims = 8
        c_z = 6
        settings = IPA_settings(dims; c_z, use_softmax1 = false)
        
        # generate random data
        L = 6
        R = 6
        B = 4
        siL = randn(Float32, dims, L, B)
        siR = siL
        zij = randn(Float32, c_z, R, L, B)
        TiL = (get_rotation(L, B), get_translation(L, B))
        TiR = TiL
        
        # Left and right equal for self attention
        @assert TiL == TiR
        @assert siL == siR
        
        # Extend the cache along both left and right
        ipa = IPCrossA(settings)
        cache = InvariantPointAttention.IPACache(settings, B)
        si = nothing
        siRs = []
        for i in 1:L
            si, cache = InvariantPointAttention.expand(ipa, cache, TiL, siL, 1, TiR, siR, 1; zij)
            push!(siRs, si)
        end
        @test cat(siRs..., dims = 2) ≈ ipa(TiL, siL, TiR, siR; zij, mask = Float32.(right_to_left_mask(6)))
    end


    @testset "IPACache_softmax1" begin
        dims = 8
        c_z = 2
        settings = IPA_settings(dims; c_z, use_softmax1 = true)
        ipa = IPCrossA(settings)

        # generate random data
        L = 5
        R = 6
        B = 4
        siL = randn(Float32, dims, L, B)
        siR = randn(Float32, dims, R, B)
        zij = randn(Float32, c_z, R, L, B)
        TiL = (get_rotation(L, B), get_translation(L, B))
        TiR = (get_rotation(R, B), get_translation(R, B))

        # check the consistency
        cache = InvariantPointAttention.IPACache(settings, B)
        siR′, cache′ = InvariantPointAttention.expand(ipa, cache, TiL, siL, L, TiR, siR, R; zij)
        @test size(siR′) == size(siR)
        @test siR′ ≈ ipa(TiL, siL, TiR, siR; zij)

        # calculate in two steps
        cache = InvariantPointAttention.IPACache(settings, B)
        siR1, cache = InvariantPointAttention.expand(ipa, cache, TiL, siL, L, TiR, siR, 2; zij)
        siR2, cache = InvariantPointAttention.expand(ipa, cache, TiL, siL, 0, TiR, siR, 4; zij)
        @test cat(siR1, siR2, dims = 2) ≈ ipa(TiL, siL, TiR, siR; zij)
    end

    @testset "IPACache_softmax1_v2" begin 
        dims = 8
        c_z = 6
        settings = IPA_settings(dims; c_z, use_softmax1 = true)
         
        # generate random data
        L = 10
        R = 10
        B = 1
        siL = randn(Float32, dims, L, B)
        siR = siL
        zij = randn(Float32, c_z, R, L, B)
        TiL = (get_rotation(L, B), get_translation(L, B))
        TiR = TiL
         
        # Left and right equal for self attention
        TiL == TiR
        siL == siR
         
        # Extend the cache along both left and right
        ipa = IPCrossA(settings)
        cache = InvariantPointAttention.IPACache(settings, B)
        siRs = []
        for i in 1:10
            si, cache = InvariantPointAttention.expand(ipa, cache, TiL, siL, 1, TiR, siR, 1; zij)
            push!(siRs, si)
        end
        @test cat(siRs..., dims = 2) ≈ ipa(TiL, siL, TiR, siR; zij, mask = right_to_left_mask(10))
    end
end