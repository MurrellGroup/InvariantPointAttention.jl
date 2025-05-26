using InvariantPointAttention
using Test

import InvariantPointAttention: get_rotation, get_translation, softmax1
import InvariantPointAttention: T_R3, T_R3_inv, pair_diff
import InvariantPointAttention: L2norm, sumabs2
import Flux

using ChainRulesTestUtils

@testset "InvariantPointAttention.jl" begin

    @testset "old_eucdists vs new_eucdists" begin
        # Test parameters
        N_head = 4
        N_query_points = 2
        N_point_values = 4
        N_frames_L = 8
        N_frames_R = 6
        batch_size = 2
        dims = 64
        c = 16

        # Create IPA settings
        settings = IPA_settings(
            dims,
            c = c,
            N_head = N_head,
            N_query_points = N_query_points,
            N_point_values = N_point_values,
            c_z = 0,
            Typ = Float32,
            use_softmax1 = false,
            scaling_qk = :default,
        )

        # Create IPA layer
        ipa = IPCrossA(settings)

        # Create test data
        TiL = (
            randn(Float32, 3, 3, N_frames_L, batch_size),  # rotation matrices
            randn(Float32, 3, 1, N_frames_L, batch_size)   # translation vectors
        )
        TiR = (
            randn(Float32, 3, 3, N_frames_R, batch_size),  # rotation matrices  
            randn(Float32, 3, 1, N_frames_R, batch_size)   # translation vectors
        )
        siL = randn(Float32, dims, N_frames_L, batch_size)
        siR = randn(Float32, dims, N_frames_R, batch_size)

        # Test with old_eucdists = true
        result_old = ipa(TiL, siL, TiR, siR; old_eucdists = true, customgrad = false)

        # Test with old_eucdists = false  
        result_new = ipa(TiL, siL, TiR, siR; old_eucdists = false, customgrad = false)

        # Check that results are approximately equal
        @test isapprox(result_old, result_new, rtol=1e-5, atol=1e-6)
        @test size(result_old) == size(result_new)
    end

    @testset "Batched rope indexing" begin
        dims = 8
        settings = IPA_settings(dims)
         
        # generate random data
        L = 10
        B = 5
        S = randn(Float32, dims, L, B)
        T = (get_rotation(L, B), get_translation(L, B))
        
        #batched residue indices
        pos_matrix = rand(1:L, L, B)
        
        ipa = IPCrossA(settings)
        rope = IPARoPE(ipa.settings.c, 100)
        
        # Use batched rope
        S1 = ipa(T, S, T, S;  mask = right_to_left_mask(10), rope = rope[pos_matrix])   
        
        # Choose custom residue ordering manually and then stack
        Tn(n) = (T[1][:,:,:,n:n], T[2][:,:,:,n:n])
        S2 = stack([
            ipa(
                Tn(n), 
                S[:,:,n:n], 
                Tn(n),
                S[:,:,n:n];
                mask = right_to_left_mask(10),
                rope = rope[pos_matrix[:,n]]
             )[:,:,1] for n in 1:B
        ])
        @test S1 ≈ S2         
    end

    @testset "Rope Expand" begin
        dims = 8
        settings = IPA_settings(dims)
         
        # generate random data
        L = 10
        R = 10
        B = 1
        siL = randn(Float32, dims, L, B)
        siR = siL
        #zij = randn(Float32, c_z, R, L, B)
        TiL = (get_rotation(L, B), get_translation(L, B))
        TiR = TiL
         
        # Left and right equal for self attention
        TiL == TiR
        siL == siR
         
        # Extend the cache along both left and right
        ipa = IPCrossA(settings)
        cache = InvariantPointAttention.IPACache(settings, B)
        
        rope = IPARoPE(ipa.settings.c, 100)
        siRs = []
        for i in 1:10
            si, cache = InvariantPointAttention.expand(ipa, cache, TiL, siL, 1, TiR, siR, 1, rope= rope.rope[i:i])
            push!(siRs, si)
        end
        cat(siRs..., dims = 2) ≈ ipa(TiL, siL, TiR, siR;  mask = right_to_left_mask(10), rope = rope[1:10])        
    end

    @testset "IPAsoftmax_invariance" begin
        batch_size = 3
        framesL = 100
        framesR = 101
        dim = 768
        
        siL = randn(Float32, dim,framesL, batch_size) 
        siR = randn(Float32, dim,framesR, batch_size)
        
        T_locL = (get_rotation(framesL, batch_size), get_translation(framesL, batch_size)) 
        T_locR = (get_rotation(framesR, batch_size), get_translation(framesR, batch_size)) 

        # Get 1 global SE(3) transformation for each batch.
        T_glob = (get_rotation(batch_size), get_translation(batch_size))
        T_GlobL = (stack([T_glob[1] for i in 1:framesL],dims = 3), stack([T_glob[2] for i in 1:framesL], dims=3))
        T_GlobR = (stack([T_glob[1] for i in 1:framesR],dims = 3), stack([T_glob[2] for i in 1:framesR], dims=3))
        
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
        @test cat(siRs..., dims = 2) ≈ ipa(TiL, siL, TiR, siR; zij, mask = right_to_left_mask(6))
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

    @testset "Softmax1" begin
        #Check if softmax1 is consistent with softmax, when adding an additional zero logit
        x = randn(4,3)
        xaug = hcat(x, zeros(4,1))
        @test softmax1(x, dims = 2) ≈ Flux.softmax(xaug, dims = 2)[:,1:end-1]
    end

    @testset "softmax1 rrule" begin
        x = randn(2,3,4)

        foreach(i -> test_rrule(softmax1, x; fkwargs=(; dims=i)), 1:3)
    end  

    @testset "T_R3 rrule" begin 
        x = randn(Float64, 3, 2, 1, 2)
        R = get_rotation(Float64, 1, 2)
        t = get_translation(Float64, 1, 2)
        test_rrule(T_R3, x, R, t)
    end

    @testset "T_R3_inv rrule" begin 
        x = randn(Float64, 3, 2, 1, 2)
        R = get_rotation(Float64, 1, 2) 
        t = get_translation(Float64, 1, 2)
        test_rrule(T_R3_inv, x, R, t)
    end

    @testset "sumabs2 rrule" begin
        x = rand(2,3,4)
        foreach(i -> test_rrule(sumabs2, x; fkwargs=(; dims=i)), 1:3)
    end

    @testset "L2norm rrule" begin 
        x = randn(2,3,4,5)
        foreach(i -> test_rrule(L2norm, x; fkwargs=(; dims=i)), 1:3)
    end

    @testset "pair_diff rrule" begin 
        x = randn(1,4,2)
        y = randn(1,3,2)
        test_rrule(pair_diff, x, y; fkwargs=(; dims=2))
    end

    @testset "ipa_customgrad" begin
        batch_size = 3
        framesL = 10
        framesR = 10
        dim = 10
        
        siL = randn(Float32, dim, framesL, batch_size) 
        siR = siL
        # Use CLOPS.jl shape notation
        TiL = (get_rotation(Float32, framesL, batch_size), get_translation(Float32, framesL, batch_size)) 
        TiR = TiL 
        zij = randn(Float32, 16, framesR, framesL, batch_size) 

        ipa = IPCrossA(IPA_settings(dim; use_softmax1 = true, c_z = 16, Typ = Float32))  
        # Batching on mask
        mask = right_to_left_mask(framesL)[:, :, ones(Int, batch_size)]
        ps = Flux.params(ipa)
        
        lz,gs = Flux.withgradient(ps) do 
            sum(ipa(TiL, siL, TiR, siR; zij, mask, customgrad = true))
        end
        
        lz2, zygotegs = Flux.withgradient(ps) do 
            sum(ipa(TiL, siL, TiR, siR; zij, mask, customgrad = false))
        end
        
        for (gs, zygotegs) in zip(keys(gs),keys(zygotegs))
            @test maximum(abs.(gs .- zygotegs)) < 2f-5
        end
        #@show lz, lz2
        @test abs.(lz - lz2) < 1f-5
    end

end