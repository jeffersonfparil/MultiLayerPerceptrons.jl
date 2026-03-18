module MultiLayerPerceptrons

using Random, LinearAlgebra, CUDA

include("helpers.jl")
export mean, var, cov, std, sampleNormal, drawreplacenot
export squarest, isfuzzymatch, levenshteindistance
include("network.jl")
export Network, init, simulate
include("activations.jl")
export linear, sigmoid, hyperbolictangent, relu, leakyrelu
export linear_derivative,
    sigmoid_derivative, hyperbolictangent_derivative, relu_derivative, leakyrelu_derivative
include("costs.jl")
export MSE, MSE_derivative, MAE, MAE_derivative, HL, HL_derivative
include("forwardbackward.jl")
export forwardpass!, backpropagation!
include("optimisers.jl")
export gradientdescent!, Adam!, AdamMax!
include("traintest.jl")
export metrics, splitdata, predict, train, optim


end
