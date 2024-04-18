using InvariantPointAttention
using InvariantPointAttention: get_rotation, get_translation
using InvariantPointAttention: softmax1
using Flux:gradient
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