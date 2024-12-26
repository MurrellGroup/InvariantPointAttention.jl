#= from AF2 supplementary: Algorithm 23 Backbone update
Takes a 3xN matrix of imaginary quaternion components, `bcd`, sets the real part to `a`, and normalizes to unit quaternions. =#
function bcds2quats(bcd::AbstractMatrix{T}, a::T=T(1)) where T<:Real
    norms = sqrt.(a .+ sum(abs2, bcd, dims=1))
    return vcat(a ./ norms, bcd ./ norms)
end

# Takes a 4xN matrix of unit quaternions and returns a 3x3xN array of rotation matrices.
function rotmatrix_from_quat(q::AbstractMatrix{<:Real})
    sx = 2q[1, :] .* q[2, :]
    sy = 2q[1, :] .* q[3, :]
    sz = 2q[1, :] .* q[4, :]

    xx = 2q[2, :] .^ 2
    xy = 2q[2, :] .* q[3, :]
    xz = 2q[2, :] .* q[4, :]

    yy = 2q[3, :] .^ 2
    yz = 2q[3, :] .* q[4, :]
    zz = 2q[4, :] .^ 2  

    r1 = 1 .- (yy .+ zz)
    r2 = xy .- sz
    r3 = xz .+ sy

    r4 = xy .+ sz
    r5 = 1 .- (xx .+ zz)
    r6 = yz .- sx

    r7 = xz .- sy
    r8 = yz .+ sx
    r9 = 1 .- (xx .+ yy)

    return reshape(vcat(r1', r4', r7', r2', r5', r8', r3', r6', r9'), 3, 3, :)
end

"""
    get_rotation([T=Float32,] dims...)

Generates random rotation matrices of given size.  
"""
get_rotation(T::Type{<:Real}, dims...) = reshape(rotmatrix_from_quat(bcds2quats(randn(T, 3, prod(dims)))), 3, 3, dims...)
get_rotation(dims...; T::Type{<:Real}=Float32) = get_rotation(T, dims...)

"""
    get_translation([T=Float32,] dims...)

Generates random translations of given size.
"""
get_translation(T::Type{<:Real}, dims...) = randn(T, 3, 1, dims...)
get_translation(dims...; T::Type{<:Real}=Float32) = get_translation(T, dims...)

# Takes a 6-dim vec and maps to a rotation matrix and translation vector, which is then applied to the input frames.
function update_frame(frames::Rigid, x::AbstractArray)
    bcds = reshape(x[:,1,:,:], 3, :)
    rotations = batchreshape(Rotation(rotmatrix_from_quat(bcds2quats(bcds))), size(x, 3), :)
    translations = Translation(x[:,2:2,:,:])
    return (translations ∘ rotations)
end

# Algorithm 21: Rigid from 3 points using the Gram–Schmidt process
function rigid_from_3points(x1::AbstractVector, x2::AbstractVector, x3::AbstractVector)
    v1 = x3 - x2
    v2 = x1 - x2
    e1 = normalize(v1)
    u2 = v2 - e1 * (e1'v2)
    e2 = normalize(u2)
    e3 = e1 × e2
    R = [e1 e2 e3]
    t = reshape(x2, 3, 1)
    return R, t
end

function get_rigid(backbone::AbstractArray{<:Real,3})
    @assert size(backbone)[1:2] == (3, 3) "first two dimensions of backbone must be 3x3"
    L = size(backbone, 3)
    R = similar(backbone, 3, 3, L)
    t = similar(backbone, 3, 1, L)
    for i in axes(backbone, 3)
        @inbounds R[:,:,i], t[:,:,i] = @views rigidFrom3points(SVector{3}(backbone[:,1,i]), SVector{3}(backbone[:,2,i]), SVector{3}(backbone[:,3,i]))
    end
    return Translation(t) ∘ Rotation(R)
end
