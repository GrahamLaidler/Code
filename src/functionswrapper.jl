using Distances,
    StaticArrays,
    Random, 
    #Distributions, #
    StatsBase, 
    ProgressMeter, 
    #MultivariateStats, # 
    Optim, 
    LineSearches, 
    LinearAlgebra,
    LatinHypercubeSampling,
    JLD2

abstract type NCAMethod end
struct NCALog <: NCAMethod end
struct NCAStandard <: NCAMethod end

include("nca.jl")
include("snca.jl")
include("functions.jl")