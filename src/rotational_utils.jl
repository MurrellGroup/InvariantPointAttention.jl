#= from AF2 supplementary: Algorithm 23 Backbone update
Takes a 3xN matrix of imaginary quaternion components, `bcd`, sets the real part to `a`, and normalizes to unit quaternions. =#
function bcds2quats(bcd::AbstractMatrix{T}, a::T=T(1)) where T<:Real
    norms = sqrt.(a .+ sum(abs2, bcd, dims=1))
    return vcat(a ./ norms, bcd ./ norms)
end
# Fast batched pairwise Euclidean distances between rows of x and columns of y
function PairwiseEuclideans(x,y)
    A_sqnorms = sum(abs2, x, dims=2)
    B_sqnorms = sum(abs2, y, dims=1)
    AB_dots = batched_mul(x,y)
    return A_sqnorms .- 2 .* AB_dots .+ B_sqnorms
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


function batched_mul_T1(x::AbstractArray{T1,N}, y::AbstractArray{T2,N}) where {T1,T2,N}
    batch_size = size(x)[3:end]
    @assert batch_size == size(y)[3:end] "batch size has to be the same for the two arrays."
    x2 = reshape(x, size(x, 1), size(x, 2), :) |> batched_transpose
    y2 = reshape(y, size(y, 1), size(y, 2), :)
    z = batched_mul(x2, y2)
    return reshape(z, size(z, 1), size(z, 2), batch_size...)
end

function batched_mul_T2(x::AbstractArray{T1,N}, y::AbstractArray{T2,N}) where {T1,T2,N}
    batch_size = size(x)[3:end]
    @assert batch_size == size(y)[3:end] "batch size has to be the same for the two arrays."
    x2 = reshape(x, size(x, 1), size(x, 2), :)
    y2 = reshape(y, size(y, 1), size(y, 2), :) |> batched_transpose
    z = batched_mul(x2, y2)
    return reshape(z, size(z, 1), size(z, 2), batch_size...)
end

#= Applies the SE3 transformations T = (rot,trans) ∈ SE(3)^N
to N batches of m points in R3, i.e., mat ∈ R^(3 x m x N) ↦ T(mat) ∈ R^(3 x m x N).
Note here that rotations here are represented in matrix form. =#
function T_R3(x::AbstractArray{T}, R::AbstractArray{T}, t::AbstractArray{T}) where T
    x′ = reshape(x, 3, size(x, 2), :)
    R′ = reshape(R, 3, 3, :)
    t′ = reshape(t, 3, 1, :)
    y′ = batched_mul(R′, x′) .+ t′
    y = reshape(y′, size(x))
    return y
end

#= Applies the group inverse of the SE3 transformations T = (R,t) ∈ SE(3)^N to N batches of m points in R3,
such that T^-1(T*x) = T^-1(Rx+t) =  R^T(Rx+t-t) = x. =#
function T_R3_inv(y::AbstractArray{T}, R::AbstractArray{T}, t::AbstractArray{T}) where T
    y′ = reshape(y, 3, size(y, 2), :)
    R′ = reshape(R, 3, 3, :)
    t′ = reshape(t, 3, 1, :)
    x′ = batched_mul(batched_transpose(R′), y′ .- t′)
    x = reshape(x′, size(y))
    return x
end

# Returns the composition of two SE(3) transformations T_1 and T_2. If T1 = (R1,t1), and T2 = (R2,t2) then T1*T2 = (R1*R2, R1*t2 + t1).
function T_T(T_1, T_2)
    R1, t1 = T_1 
    R2, t2 = T_2
    new_rot = batched_mul(R1,R2)
    new_trans = reshape(batched_mul(R1,t2), size(t1)) .+ t1
    return (new_rot,new_trans)
end

# Takes a 6-dim vec and maps to a rotation matrix and translation vector, which is then applied to the input frames.
function update_frame(Ti, arr)
    bcds = reshape(arr[:,1,:,:],3,:)
    rotmat = rotmatrix_from_quat(bcds2quats(bcds))  
    T_new = (
        reshape(rotmat,3,3,size(arr,3),:),
        reshape(arr[:,2,:,:],3,1,size(arr,3),:)
            )
    T = T_T(Ti,T_new)
    return T
end

unzip(a) = map(x->getfield.(a, x), fieldnames(eltype(a)))

centroid(coords::AbstractMatrix) = vec(sum(coords; dims=2)) / size(coords, 2)

# Get rotation matrix and translation from residue
function calculate_residue_rotation_and_translation(residue_xyz::AbstractMatrix)
    # Returns the rotation matrix and the translation of a gien residue. 
    N = residue_xyz[:, 1]
    Ca = residue_xyz[:, 2] # We use the centroid instead of the Ca - not 100% sure if this is correct
    C = residue_xyz[:, 3]

    t = centroid(residue_xyz)

    v1 = C - t
    v2 = N - t
    e1 = normalize(v1)
    u2 = v2 - e1 * (e1'v2)
    e2 = normalize(u2)
    e3 = cross(e1,e2)
    R = hcat(e1, e2, e3)
    return R, t
end

"""
    get_T(coords::Array{<:Real, 3})
Get the assosciated SE(3) frame for all residues in a protein backbone represented as a 3x3xL array of coordinates.
"""
function get_T(coords::Array{<:Real, 3})
    Ti = stack.(unzip([calculate_residue_rotation_and_translation(coords[:,:,i]) for i in axes(coords,3)]))
    return (Ti[1],reshape(Ti[2],3,1,:))
end

"""
Get the associated SE(3) frames for all residues in a batch of proteins
"""
function get_T_batch(coords::Array{<:Real, 4})
    rots = zeros(3,3,size(coords)[3:4]...)
    trans = zeros(3,1,size(coords)[3:4]...)
    for j in axes(coords,4)
        Tij = get_T(coords[:,:,:,j])
        rots[:,:,:,j] = Tij[1]
        trans[:,:,:,j] = Tij[2]
    end
    return (rots, trans)
end

# Index into a T up to index i.
function T_till(T,i)
    Tr, Tt = T[1][:,:,1:i,:], T[2][:,:,1:i,:]
    return Tr, Tt
end
