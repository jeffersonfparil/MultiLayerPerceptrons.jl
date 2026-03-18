"""
Computes the mean of a matrix along the specified dimension.

# Arguments
- `X::Matrix{T}`: Input matrix.
- `dims::Int64=1`: Dimension to compute the mean over (default: 1)).

# Returns
- `Matrix{T}`: Mean values along the specified dimension.

# Examples
```jldoctest; setup = :(using MultiLayerPerceptrons)
julia> X = rand(10, 42);

julia> sum(mean(X, dims=1) .> 0.0) == 42
true

julia> sum(mean(X, dims=2) .> 0.0) == 10
true

julia> Y = zeros(10, 42);

julia> sum(mean(Y, dims=1) .== 0.0) == 42
true

julia> sum(mean(Y, dims=2) .== 0.0) == 10
true
```
"""
function mean(X::Matrix{T}; dims::Int64 = 1)::Matrix{T} where {T<:AbstractFloat}
    # X = rand(10, 4)
    dims > 2 ? throw("Maximum `dims` is 2 because `X` is a matrix, i.e. 2-dimensional") :
    nothing
    n = size(X, dims)
    sum(X, dims = dims) ./ n
end


"""
Computes the sample variance of a matrix along the specified dimension.

# Arguments
- `X::Matrix{T}`: Input matrix.
- `dims::Int64=1`: Dimension to compute the variance over (default: 1)).

# Returns
- `Matrix{T}`: Variance values along the specified dimension.

# Examples
```jldoctest; setup = :(using MultiLayerPerceptrons)
julia> X = rand(10, 42);

julia> sum(var(X, dims=1) .> 0.0) == 42
true

julia> sum(var(X, dims=2) .> 0.0) == 10
true

julia> Y = zeros(10, 42);

julia> sum(var(Y, dims=1) .== 0.0) == 42
true

julia> sum(var(Y, dims=2) .== 0.0) == 10
true
```
"""
function var(X::Matrix{T}; dims::Int64 = 1)::Matrix{T} where {T<:AbstractFloat}
    # X = rand(10, 4)
    dims > 2 ? throw("Maximum `dims` is 2 because `X` is a matrix, i.e. 2-dimensional") :
    nothing
    n = size(X, dims)
    μ = sum(X, dims = dims) ./ n
    sum((X .- μ) .^ 2, dims = dims) ./ (n - 1)
end


"""
Computes the sample standard deviation of a matrix along the specified dimension.

# Arguments
- `X::Matrix{T}`: Input matrix.
- `dims::Int64=1`: Dimension to compute the standard deviation over (default: 1)).

# Returns
- `Matrix{T}`: Standard deviation values.

# Examples
```jldoctest; setup = :(using MultiLayerPerceptrons)
julia> X = rand(10, 42);

julia> sum(std(X, dims=1) .> 0.0) == 42
true

julia> sum(std(X, dims=2) .> 0.0) == 10
true

julia> Y = zeros(10, 42);

julia> sum(std(Y, dims=1) .== 0.0) == 42
true

julia> sum(std(Y, dims=2) .== 0.0) == 10
true
```
"""
function std(X::Matrix{T}; dims::Int64 = 1)::Matrix{T} where {T<:AbstractFloat}
    sqrt.(var(X, dims = dims))
end


"""
Computes the sample covariance between two matrices along the specified dimension.

# Arguments
- `X::Matrix{T}`: First input matrix.
- `Y::Matrix{T}`: Second input matrix.
- `dims::Int64=1`: Dimension to compute the covariance over (default: 1)).

# Returns
- `Matrix

# Examples
```jldoctest; setup = :(using MultiLayerPerceptrons)
julia> X = rand(10, 42);

julia> Y = rand(10, 42);

julia> sum(cov(X, Y) .!= 0.0) == 42
true
```
"""
function cov(
    X::Matrix{T},
    Y::Matrix{T};
    dims::Int64 = 1,
)::Matrix{T} where {T<:AbstractFloat}
    # X = rand(10, 4); Y = rand(10, 4); dims = 2
    size(X) != size(Y) ? throw("The dimensions of the input matrices are not the same") :
    nothing
    dims > 2 ? throw("Maximum `dims` is 2 because `X` is a matrix, i.e. 2-dimensional") :
    nothing
    n = size(X, dims)
    μ_X = sum(X, dims = dims) ./ n
    μ_Y = sum(Y, dims = dims) ./ n
    sum((X .- μ_X) .* (Y .- μ_Y), dims = dims) ./ (n - 1)
end


"""
Generates samples from a standard normal distribution using the Box-Muller transform.

# Arguments
- `n::Int64`: Number of samples.
- `μ::T=0.0`: Mean of the distribution (default: 0.0).
- `σ::T=1.0`: Standard deviation of the distribution (default: 1.0).

# Returns
- `Vector{T}`: Vector of normally distributed samples.

# Examples
```jldoctest; setup = :(using MultiLayerPerceptrons)
julia> x = sampleNormal(1_000, μ=0.0,σ=1.00);

julia> abs(mean(hcat(x), dims=1)[1,1]) < 0.1
true

julia> abs(1.00 - std(hcat(x), dims=1)[1,1]) < 0.1
true
```
"""
function sampleNormal(
    n::Int64;
    μ::T = 0.0,
    σ::T = 1.00,
    seed::Int64 = 42,
)::Vector{T} where {T<:AbstractFloat}
    # n = 100; μ = 0.0; σ = 1.0;
    rng = Random.seed!(seed)
    U = rand(rng, n)
    V = rand(rng, n)
    z = sqrt.(-2.00 .* log.(U)) .* cos.(2.00 * π .* V)
    T.((z .* σ) .+ μ)
end


"""
Draws `n` unique samples from `N` elements without replacement.

# Arguments
- `N::T`: Total number of elements.
- `n::T`: Number of samples to draw.

# Returns
- `Vector{T}`: Vector of unique sampled indices.

# Examples
```jldoctest; setup = :(using MultiLayerPerceptrons)
julia> x = drawreplacenot(100, 100);

julia> length(x) == 100
true

julia> sum(x) == 101*50
true

julia> X = [drawreplacenot(100, 10) for _ in 1:1_000];

julia> unique([length(x) for x in X]) == [10]
true

julia> abs(50.0 - mean(hcat([mean(hcat(Float64.(x)), dims=1)[1,1] for x in X]))[1,1]) < 10.0
true
```
"""
function drawreplacenot(N::T, n::T)::Vector{T} where {T<:Integer}
    # N=1_000; n=12; T = Int64
    if N < n
        throw("Cannot draw $n samples from $N total elements.")
    end
    idx::Vector{T} = []
    m = n
    while m > 0
        append!(idx, T.(ceil.(N * rand(Float64, m))))
        idx = unique(idx)
        m = n - length(idx)
    end
    idx
end


"""
    squarest(N::Int64)::Tuple{Int64, Int64}

Calculate the most square-like dimensions for arranging N elements in a grid.

Takes an integer N and returns a tuple of two integers representing the number of rows
and columns that would create the most square-like arrangement possible while exactly
fitting N elements.

# Arguments
- `N::Int64`: Total number of elements to arrange in a grid

# Returns
- `Tuple{Int64, Int64}`: A tuple containing (number of rows, number of columns)

# Examples
```jldoctest; setup = :(using MultiLayerPerceptrons)
julia> squarest(9)
(3, 3)

julia> squarest(12)
(3, 4)

julia> squarest(15)
(3, 5)

julia> squarest(500)
(20, 25)
```
"""
function squarest(N::Int64)::Tuple{Int64,Int64}
    selections = collect(1:Int(ceil(sqrt(N))))
    n_rows = selections[[N%x for x in selections] .== 0.0][end]
    n_cols = Int(N / n_rows)
    out = sort([n_rows, n_cols])
    (out[1], out[2])
end

"""
    levenshteindistance(a::String, b::String)::Int64

Calculate the Levenshtein distance (edit distance) between two strings.

The Levenshtein distance is a measure of the minimum number of single-character edits 
(insertions, deletions, or substitutions) required to change one string into another.

# Arguments
- `a::String`: First input string
- `b::String`: Second input string

# Returns
- `Int64`: The minimum number of edits needed to transform string `a` into string `b`

# Examples
```jldoctest; setup = :(using MultiLayerPerceptrons)
julia> levenshteindistance("populations", "populations")
0

julia> levenshteindistance("populations", "poplation")
2

julia> levenshteindistance("populations", "entry")
3
```
"""
function levenshteindistance(a::String, b::String)::Int64
    # a = "populations"; b = "entry";
    n::Int64 = length(a)
    m::Int64 = length(b)
    d::Matrix{Int64} = fill(0, n, m)
    for j = 2:m
        d[1, j] = j - 1
    end
    cost::Int64 = 0
    deletion::Int64 = 0
    insertion::Int64 = 0
    substitution::Int64 = 0
    for j = 2:m
        for i = 2:n
            if a[i] == b[j]
                cost = 0
            else
                cost = 1
            end
            deletion = d[i-1, j] + 1
            insertion = d[i, j-1] + 1
            substitution = d[i-1, j-1] + cost
            d[i, j] = minimum([deletion, insertion, substitution])
        end
    end
    d[end, end]
end

"""
    isfuzzymatch(a::String, b::String; threshold::Float64=0.3)::Bool

Determines if two strings approximately match each other using Levenshtein distance.

The function compares two strings and returns `true` if they are considered similar enough
based on the Levenshtein edit distance and a threshold value. The threshold is applied as
a fraction of the length of the shorter string. Additionally, the function normalizes specific
string inputs (e.g., `"#chr"` is replaced with `"chrom"`) before comparison.

# Arguments
- `a::String`: First string to compare
- `b::String`: Second string to compare
- `threshold::Float64=0.3`: Maximum allowed edit distance as a fraction of the shorter string length

# Returns
- `Bool`: `true` if the strings match within the threshold, `false` otherwise

# Examples
```jldoctest; setup = :(using MultiLayerPerceptrons)
julia> isfuzzymatch("populations", "populations")
true

julia> isfuzzymatch("populations", "poplation")
true

julia> isfuzzymatch("populations", "entry")
false
```
"""
function isfuzzymatch(a::String, b::String; threshold::Float64 = 0.3)::Bool
    # a = "populations"; b = "populatins"; threshold = 0.3;
    a = if (a == "#chr")
        "chrom"
    else
        a
    end
    b = if (b == "#chr")
        "chrom"
    else
        b
    end
    n::Int64 = length(a)
    m::Int64 = length(b)
    dist::Int64 = levenshteindistance(a, b)
    dist < Int64(round(minimum([n, m]) * threshold))
end
