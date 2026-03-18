"""
A struct representing a feed-forward neural network.

# Fields
- `n_hidden_layers::Int64`: Number of hidden layers.
- `n_hidden_nodes::Vector{Int64}`: Number of nodes in each hidden layer.
- `dropout_rates::Vector{T}`: Dropout rates for each hidden layer.
- `W::Vector{CuArray{T, 2}}`: Weight matrices for each layer.
- `b::Vector{CuArray{T, 1}}`: Bias vectors for each layer.
- `ŷ::CuArray{T, 2}`: Output predictions of the network.
- `S::Vector{CuArray{T, 2}}`: Pre-activation values (weighted sums).
- `A::Vector{CuArray{T, 2}}`: Activations for each layer including input.
- `∇W::Vector{CuArray{T, 2}}`: Gradients of weights.
- `∇b::Vector{CuArray{T, 1}}`: Gradients of biases.
- `F::Function`: Activation function.
- `∂F::Function`: Derivative of the activation function.
- `C::Function`: Cost function.
- `∂C::Function`: Derivative of the cost function.
- `seed::Int64`: Random seed for reproducibility.

# Note
See `init(::CuArray{T,2}; ...)` function for network initialisation.
"""
struct Network{T}
    n_hidden_layers::Int64 # number of hidden layers
    n_hidden_nodes::Vector{Int64} # number of nodes per hidden layer
    dropout_rates::Vector{T} # soft dropout rates per hidden layer
    W::Vector{CuArray{T,2}} # weights
    b::Vector{CuArray{T,1}} # biases
    ŷ::CuArray{T,2} # predictions
    S::Vector{CuArray{T,2}} # summed weights (i.e. prior to activation function)
    A::Vector{CuArray{T,2}} # activation function output including the input layer as the first element
    ∇W::Vector{CuArray{T,2}} # gradients of the weights
    ∇b::Vector{CuArray{T,1}} # gradients of the biases
    F::Function # activation function
    ∂F::Function # derivative of the activation function
    C::Function # cost function
    ∂C::Function # derivative of the cost function
    seed::Int64 # random seed for dropouts
end


"""
    init(
        X::CuArray{T,2};
        n_hidden_layers::Int64 = 2,
        n_hidden_nodes::Vector{Int64} = repeat([256], n_hidden_layers),
        dropout_rates::Vector{Float64} = repeat([0.0], n_hidden_layers),
        F::Function = relu,
        ∂F::Function = relu_derivative,
        C::Function = MSE,
        ∂C::Function = MSE_derivative,
        y::Union{Nothing,CuArray{T,2}} = nothing,
        seed::Int64 = 42,
    )::Network{T} where {T<:AbstractFloat}

Initializes a neural network with specified architecture and parameters.

# Arguments
- `X::CuArray{T, 2}`: Input data.
- `n_hidden_layers::Int64=2`: Number of hidden layers (default: 2).
- `n_hidden_nodes::Vector{Int64}=repeat([256], n_hidden_layers)`: Nodes per hidden layer (default: [256, 256]).
- `dropout_rates::Vector{Float64}=repeat([0.0], n_hidden_layers)`: Dropout rates per layer (default: [0.0, 0.0]).
- `F::Function=relu`: Activation function (default: `relu`).
- `∂F::Function=relu_derivative`: Derivative of activation function (default: `relu_derivative`).
- `C::Function=MSE`: Cost function (default: `MSE`).
- `∂C::Function=MSE_derivative`: Derivative of cost function (default: `MSE_derivative`).
- `y::Union{Nothing, CuArray{T,2}}=nothing`: Optional target values for OLS initialisation.
- `seed::Int64=42`: Random seed (default: 42).

# Notes:
- Assumes the output layer has a single node (1-dimensional)

# Returns
- `Network{T}`: Initialized network.

# Examples
```jldoctest; setup = :(using MultiLayerPerceptrons, CUDA)
julia> X = CuArray(reshape(sampleNormal(100_000), (100, 1_000)));

julia> Ω = init(X, n_hidden_layers=5, n_hidden_nodes=repeat([128], 5), F=leakyrelu, ∂F=leakyrelu_derivative);

julia> (Ω.n_hidden_layers == 5) && (Ω.n_hidden_nodes == repeat([128], 5)) && (Ω.F == leakyrelu) && (Ω.∂F == leakyrelu_derivative)
true
```
"""
function init(
    X::CuArray{T,2};
    n_hidden_layers::Int64 = 2,
    n_hidden_nodes::Vector{Int64} = repeat([256], n_hidden_layers),
    dropout_rates::Vector{Float64} = repeat([0.0], n_hidden_layers),
    F::Function = relu,
    ∂F::Function = relu_derivative,
    C::Function = MSE,
    ∂C::Function = MSE_derivative,
    y::Union{Nothing,CuArray{T,2}} = nothing,
    seed::Int64 = 42,
)::Network{T} where {T<:AbstractFloat}
    # T = Float32; X = CUDA.randn(1_000, 100); n_hidden_layers::Int64=2; n_hidden_nodes::Vector{Int64}=repeat([256], n_hidden_layers); dropout_rates::Vector{Float64}=repeat([0.0], n_hidden_layers); F::Function=relu; ∂F::Function=relu_derivative; C::Function=MSE; ∂C::Function=MSE_derivative; seed::Int64 = 42
    # T = Float32; X = CUDA.randn(1_000, 100); n_hidden_layers::Int64=0; n_hidden_nodes::Vector{Int64}=repeat([0], n_hidden_layers); dropout_rates::Vector{Float64}=repeat([0.0], n_hidden_layers); F::Function=relu; ∂F::Function=relu_derivative; C::Function=MSE; ∂C::Function=MSE_derivative; seed::Int64 = 42
    # y::Union{Nothing, CuArray{T,2}} = nothing
    # y::Union{Nothing, CuArray{T,2}} = CUDA.randn(1, 100)
    Random.seed!(seed)
    p, n = size(X)
    n_total_layers = 1 + n_hidden_layers + 1 # input layer + hidden layers + output layer
    n_total_nodes = vcat(collect(p), n_hidden_nodes, 1)
    W = []
    b = []
    for i = 1:(n_total_layers-1)
        # i = 1
        n_i = n_total_nodes[i+1]
        p_i = n_total_nodes[i]
        # Initialise the weights as: w ~ N(0, 0.1) and all biases to zero
        w = if !isnothing(y) && (i == 1)
            # Force the first layer to have the OLS solution
            β = pinv(X * X') * (X*y')[:, 1]
            CuArray{T,2}(reshape(repeat(β, outer = n_i), p_i, n_i)')
        else
            CuArray{T,2}(reshape(sampleNormal(n_i * p_i, μ = T(0.0), σ = T(0.1)), n_i, p_i))
        end
        push!(W, w)
        push!(b, CuArray{T,1}(zeros(n_i)))
        # UnicodePlots.histogram(reshape(Matrix(W[1]), n_i*p_i))
    end
    # [size(x) for x in W]
    # [size(x) for x in b]
    ŷ = CuArray{T,2}(zeros(1, n))
    A = vcat(
        [deepcopy(X)],
        [CuArray{T,2}(zeros(size(W[i], 2), n)) for i = 2:(n_total_layers-1)],
    )
    # [size(x) for x in A]
    S = vcat([
        CuArray{T,2}(zeros(size(W[i], 1), size(A[i], 2))) for i = 1:(n_total_layers-1)
    ])
    # [size(x) for x in S]
    ∇W = [CuArray{T,2}(zeros(size(x))) for x in W]
    ∇b = [CuArray{T,1}(zeros(size(x))) for x in b]
    Network{T}(
        n_hidden_layers,
        n_hidden_nodes,
        T.(dropout_rates),
        W,
        b,
        ŷ,
        S,
        A,
        ∇W,
        ∇b,
        F,
        ∂F,
        C,
        ∂C,
        seed,
    )
end


"""
Simulates input and output data for training a neural network.

# Keyword Arguments
- `T::Type=Float32`: Data type (default: `Float32`).
- `seed::Int64=42`: Random seed (default: 42).
- `n::Int64=1000`: Number of samples (default: 1000).
- `p::Int64=10`: Number of features (default: 10).
- `l::Int64=5`: Number of hidden layers (default: 5).
- `d::Float64=0.00`: Dropout rate (default: 0.00).
- `F::Function=relu`: Activation function (default: `relu`).
- `phen_type::String="non-linear"`: Phenotype type ("random", "linear", "non-linear"; default: "non-linear").
- `normalise_X::Bool=true`: Whether to normalize input features (default: true).
- `normalise_y::Bool=true`: Whether to normalize output (default: true).
- `h²::Float64=0.75`: Heritability for phenotype generation (default: 0.75).
- `p_unobserved::Int64=1`: Number of unobserved features (default: 1).

# Returns
- `Dict{String, CuArray{T, 2}}`: Dictionary containing input `X` and output `y`.

# Examples
```jldoctest; setup = :(using MultiLayerPerceptrons, CUDA)
julia> Xy = simulate(n=123);

julia> ŷ = CuArray(Float32.(reshape(sampleNormal(123, seed=456), (1, 123))));

julia> M = metrics(ŷ, Xy["y"]);

julia> (abs(M["ρ"]) < 0.5) && (M["R²"] < 0.5)
true
```
"""
function simulate(;
    T::Type = Float32,
    seed::Int64 = 42,
    n::Int64 = 1_000,
    p::Int64 = 10,
    l::Int64 = 5,
    d::Float64 = 0.00,
    F::Function = relu,
    phen_type::String = ["random", "linear", "non-linear"][3],
    normalise_X::Bool = true,
    normalise_y::Bool = true,
    h²::Float64 = 0.75,
    p_unobserved::Int64 = 1,
)::Dict{String,CuArray{T,2}}
    # T=Float32; seed = 42; n = 1_000; p = 10; l = 5; d = 0.00; phen_type = "non-linear"; p_unobserved = 1; h² = 0.75; normalise_X = true; normalise_y = true; F::Function = relu
    Random.seed!(seed)
    CUDA.seed!(seed)
    X = rand(Bool, n, p)
    ϕ = if phen_type == "random"
        rand(n, 1)
    elseif phen_type == "linear"
        β = rand(p, 1)
        σ²g = var(X * β, dims = 1)[1, 1]
        σ²e = σ²g * ((1 / h²) - 1.00)
        e = sampleNormal(n, μ = T(0.00), σ = T(sqrt(σ²e)))
        (X * β) .+ e
    elseif phen_type == "non-linear"
        X_true = CuArray{T,2}(Matrix(hcat(X, rand(Bool, n, p_unobserved))'))
        Ω = init(
            X_true,
            n_hidden_layers = l,
            F = F,
            dropout_rates = repeat([d], l),
            seed = seed,
        )
        for i = 1:Ω.n_hidden_layers
            # i = 1
            Ω.W[i] .= CUDA.randn(size(Ω.W[i], 1), size(Ω.W[i], 2))
            Ω.b[i] .= CUDA.randn(size(Ω.b[i], 1))
        end
        forwardpass!(Ω)
        Matrix(Ω.ŷ')
    else
        throw(
            "Please use `phen_type=\"random\"` or `phen_type=\"linear\"` or `phen_type=\"non-linear\"`",
        )
    end
    X = normalise_X ? (X .- mean(T.(X), dims = 1)) ./ std(T.(X), dims = 1) : X
    ϕ = normalise_y ? (ϕ .- mean(T.(ϕ), dims = 1)) ./ std(T.(ϕ), dims = 1) : X
    X = CuArray{T,2}(Matrix(X'))
    y = CuArray{T,2}(reshape(ϕ, (1, n)))
    Dict("X" => X, "y" => y)
end
