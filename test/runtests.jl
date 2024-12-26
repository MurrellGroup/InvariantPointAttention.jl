using InvariantPointAttention
using Test

using Flux

using BatchedTransformations
using ChainRulesTestUtils

@testset "InvariantPointAttention.jl" begin

    @testset "IPA" begin

        @testset for (softmax, n_dims_z, use_mask) in Iterators.product([softmax1, softmax], [0, 16], [true, false])

            n_dims_s = 256
            n_frames_Q = 20
            n_frames_K = 21
            batch_size = 3

            s_Q = randn(Float32, n_dims_s, n_frames_Q, batch_size)
            s_K = randn(Float32, n_dims_s, n_frames_K, batch_size)
            frames_Q = rand(Float32, Rigid, 3, (n_frames_Q, batch_size))
            frames_K = rand(Float32, Rigid, 3, (n_frames_K, batch_size))

            global_frames = rand(Float32, Rigid, 3, (1, batch_size))
            frames_rotated_Q = batchrepeat(global_frames, n_frames_Q) ∘ frames_Q
            frames_rotated_K = batchrepeat(global_frames, n_frames_K) ∘ frames_K

            ipa = StructureModule(IPAConfig(; n_dims_s, n_dims_z, softmax))

            @testset "cross" begin
                z = iszero(n_dims_z) ? nothing : randn(Float32, n_dims_z, n_frames_Q, n_frames_K, batch_size)
                mask = use_mask ? .!rand(Bool, n_frames_Q, n_frames_K, batch_size) .* -Inf32 : nothing
                new_frames_Q,  new_s_Q  = ipa(frames_Q, s_Q, frames_K, s_K, z; mask)
                new_frames_rotated_Q, new_s_rotated_Q = ipa(frames_rotated_Q, s_Q, frames_rotated_K, s_K, z; mask)
                @test new_s_rotated_Q ≈ new_s_Q
            end

            @testset "self" begin
                z = iszero(n_dims_z) ? nothing : randn(Float32, n_dims_z, n_frames_Q, n_frames_Q, batch_size)
                mask = use_mask ? .!rand(Bool, n_frames_Q, n_frames_Q, batch_size) .* -Inf32 : nothing
                new_frames_Q,  new_s_Q  = ipa(frames_Q, s_Q, frames_Q, s_Q, z; mask)
                new_frames_rotated_Q, new_s_rotated_Q = ipa(frames_rotated_Q, s_Q, frames_rotated_Q, s_Q, z; mask)
                @test new_s_rotated_Q ≈ new_s_Q
            end

        end

    end

    #=@testset "IPACache" begin
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
    end=#


    #=@testset "IPACache_softmax1" begin
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
    end=#

    # Check if softmax1 is consistent with softmax, when adding an additional zero logit
    @testset "Softmax1" begin
        x = randn(4,3)
        xaug = hcat(x, zeros(4,1))
        @test softmax1(x, dims=2) ≈ Flux.softmax(xaug, dims=2)[:,1:end-1]
    end

    @testset "softmax1 rrule" begin
        x = randn(2,3,4)
        foreach(i -> test_rrule(softmax1, x; fkwargs=(; dims=i)), 1:3)
    end  

    #=@testset "ipa_customgrad" begin
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
    end=#

end