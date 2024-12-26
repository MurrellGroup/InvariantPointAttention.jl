# See https://youtu.be/lT6cRbjD1C8?t=1333

"""
    softmax1(x, dims = 1)

Behaves like softmax, but as though there was an additional logit of zero along
dims (which is excluded from the output). So the values will sum to a value
between zero and 1.

See https://www.evanmiller.org/attention-is-off-by-one.html
"""
function softmax1(x::AbstractArray{T}; dims = 1) where {T}
    _zero = T(0)
    max_ = max.(fast_maximum2(x; dims), _zero)
    @fastmath out = exp.(x .- max_)
    tmp = sum(out, dims = dims)
    out ./ (tmp + exp.(-max_))
end

# taken from NNlib
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