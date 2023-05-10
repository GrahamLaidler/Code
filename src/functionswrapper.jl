using Distances,
    StaticArrays,
    Random, 
    StatsBase, 
    ProgressMeter, 
    Optim, 
    LineSearches, 
    LinearAlgebra,
    LatinHypercubeSampling,
    JLD2,
    Plots.PlotMeasures,
    LaTeXStrings

abstract type NCAMethod end
struct NCALog <: NCAMethod end
struct NCAStandard <: NCAMethod end

include("nca.jl")
include("snca.jl")
include("functions.jl")