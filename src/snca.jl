"""
    SNCA(A, x, y; objective)

Compute the SNCA objective function for matrix A with data x and y, using defined objective scaling.

# Arguments:

- `A` should be a `SMatrix` with dimensions `P` and `D`. When optimising SNCA, `A` should be a Matrix, so we include another SNCA function to accommodate this.
- `x` should be an `AbstractVector` of `SVector`s, each of which is a point in the space.
- `y` should be an `AbstractVector` containing the classes of the `x` points.
- `objective` is a named argument which whould be of type `NCAMethod`. Choose from:
    - `NCAStandard()` for standard (which is the default).
    - `NCALog()` for log.

If optimising SNCA, use the following:
- `A` should be a Matrix.
- `x` should be an `AbstractVector` of `SVector`s, each of which is a point in the space.
- `y` should be an `AbstractVector` containing the classes of the `x` points.
- `objective` is a named argument which should be of type `NCAMethod`. Choose from:
    - `NCAStandard()` for standard (which is the default).
    - `NCALog()` for log.
- `dims` is an additional named argument which should be `Val(P)` where `P = size(A, 1)`, the solution dimension.

# Examples:
```julia
julia> A = SMatrix{2,2,Float64}([1.0 2; 3 4]) # generate A

julia> x_matrix = rand(MvNormal(ones(2),[1 0; 0 1]),100) # generate matrix of x values

julia> x = svectorscopy(x_matrix, Val(2)) # convert to vector of SVectors

julia> y = rand(Bernoulli(0.5),100) # generate y

julia> SNCA(A,x,y) # run standard

julia> SNCA(A,x,y,objective = NCALog()) # run log

julia> A_initial = [1.0 0; 0 1] # generate initial A for optimisation

julia> optimize(A -> SNCA(A, x, y, dims = Val(2)), A_initial, LBFGS())
```

"""

function SNCA(A, x::AbstractVector{SVector{D,T}}, y::AbstractVector; objective=NCAStandard(), dims::Val{P}) where {P,D,T,L}
    A = SMatrix{P,D,T}(A)
    SNCA(A, x, y; objective=objective)
end
function SNCA(A::SMatrix{P,D,T,L}, x::AbstractVector{SVector{D,T}}, y::AbstractVector; objective::NCAMethod=NCAStandard()) where {P,D,T,L}
    length(x)==length(y) || throw(ArgumentError("x and y should be of the same length."))
    joint = [(xᵢ,yᵢ) for (xᵢ,yᵢ) ∈ zip(x,y)]
    cells = countmap(joint)
    n = length(y)
    d = SqEuclidean()
    value = 0.0
    distances = Vector{T}(undef, length(cells))
    for kᵢ in keys(cells)
        for (j,kⱼ) ∈ enumerate(keys(cells))
            distances[j] = d(A*kᵢ[1], A*kⱼ[1])
        end
        distances .-= minimum(distances[j] + Inf * (kᵢ[1] == kⱼ[1]) for (j,kⱼ) ∈ enumerate(keys(cells))) 
        pᵢ = zero(eltype(distances))
        totalᵢ = zero(eltype(distances))
        for (j,kⱼ) ∈ enumerate(keys(cells))
            tmp = exp(-distances[j]) * cells[kⱼ] * (kⱼ[1] != kᵢ[1]) 
            pᵢ += tmp * (kⱼ[2] == kᵢ[2]) 
            totalᵢ += tmp
        end
        if objective isa NCAStandard
            value += pᵢ/totalᵢ * cells[kᵢ]
        elseif objective isa NCALog
            value += (log(pᵢ)-log(totalᵢ)) * cells[kᵢ]
        end
    end
    return -value/n
end

"""
    SNCAincremental(A, A_fixed, x, aggP, y; objective)

Compute the SNCA objective function for use in the SNCA optimisation algorithm, using defined objective scaling.
The projection matrix is split into rows `A_fixed` and `A`. 
The distance contribution from `A_fixed` is calculated from `A_fixed * x`.
The distance contribution from `A` is calculated from `A * aggP * x`.
In the optimisation algorithm, aggP is chosen to represent a projection into the nullspace of A_fixed, in order that the optimised A will be orthogonal to the existing rows of A_fixed. 

# Arguments:

- `A` should be a `SMatrix` with dimensions `P` and `D`.
- `A_fixed` should be a Matrix.
- `x` should be an `AbstractVector` of `SVector`s, each of which is a point in the space.
- `aggP` should be a Matrix.
- `y` should be an `AbstractVector` containing the classes of the `x` points.
- `objective` is a named argument which whould be of type `NCAMethod`. Choose from:
    - `NCAStandard()` for standard (which is the default).
    - `NCALog()` for log.

If optimising SNCAincremental (optimising over `A`), use the following:
- `A` should be a Matrix.
- `A_fixed` should be a Matrix.
- `x` should be an `AbstractVector` of `SVector`s, each of which is a point in the space.
- `aggP` should be a Matrix.
- `y` should be an `AbstractVector` containing the classes of the `x` points.
- `objective` is a named argument which should be of type `NCAMethod`. Choose from:
    - `NCAStandard()` for standard (which is the default).
    - `NCALog()` for log.
- `dims` is an additional named argument which should be `Val(P)` where `P = size(A, 1)`, the solution dimension. When optimising SNCA row by row, `dims` will always be `Val(1)`.
```

"""

function SNCAincremental(A, A_fixed, x::AbstractVector{SVector{D,T}}, aggP, y::AbstractVector; objective=NCAStandard(), dims::Val{P}) where {P,D,T,L}
    A = SMatrix{P,D,T}(A)
    SNCAincremental(A, A_fixed, x, aggP, y; objective=objective)
end
function SNCAincremental(A::SMatrix{P,D,T,L}, A_fixed, x::AbstractVector{SVector{D,T}}, aggP, y::AbstractVector; objective::NCAMethod=NCAStandard()) where {P,D,T,L}
    joint = [(xᵢ,yᵢ) for (xᵢ,yᵢ) ∈ zip(x,y)]
    cells = countmap(joint)
    n = length(y)
    d = SqEuclidean()
    value = 0.0
    distances = Vector{T}(undef, length(cells))
    currentprojs = Vector{Vector{T}}(undef, length(cells))
    newprojs = Vector{Vector{T}}(undef, length(cells))
    for (i,kᵢ) in enumerate(keys(cells))
        currentprojs[i] = A_fixed*kᵢ[1]
        newprojs[i] = A*aggP*kᵢ[1]
    end
    for (i,kᵢ) in enumerate(keys(cells))
        for (j,kⱼ) ∈ enumerate(keys(cells))
            distances[j] = d(newprojs[i], newprojs[j]) + d(currentprojs[i], currentprojs[j])
        end
        distances .-= minimum(distances[j] + Inf * (kᵢ[1] == kⱼ[1]) for (j,kⱼ) ∈ enumerate(keys(cells))) 
        pᵢ = zero(eltype(distances))
        totalᵢ = zero(eltype(distances))
        for (j,kⱼ) ∈ enumerate(keys(cells))
            tmp = exp(-distances[j]) * cells[kⱼ] * (kⱼ[1] != kᵢ[1]) 
            pᵢ += tmp * (kⱼ[2] == kᵢ[2]) 
            totalᵢ += tmp
        end
        if objective isa NCAStandard
            value += pᵢ/totalᵢ * cells[kᵢ]
        elseif objective isa NCALog
            value += (log(pᵢ)-log(totalᵢ)) * cells[kᵢ]
        end
    end
    return -value/n
end

"""
    initLHS(vars; n, style, nrows)

Generate a set of initialisations for optimisation, using a Latin Hypercube space-filling design. 

# Arguments:

- `vars` is the number of elements in an initialisation. This is the number of elements in the projection matrix A to be optimised.
- `n` is the number of initialisation desired (the number of points in the Latin Hypercube design).
- `style` is a named argument which whould be of type `initMethod`. Choose from:
    - `initRow()` if initialising row matrices (which is the default).
    - `initMatrix()` if initialising a Matrix with more than one row.
- `nrows` is the number of rows in the initialisations. The default is 1. If the method `initMatrix()` is used, `nrows` should be set.

"""

abstract type initMethod end
struct initRow <: initMethod end
struct initMatrix <: initMethod end
    
function initsLHS(vars; n, style::initMethod=initRow(), nrows=1)
    isinteger(vars/nrows) || throw(ArgumentError("vars should be an integer multiple of nrows"))
    initializations = Array{Array{Float64, 2}, 1}(undef, n)
    plan, _ = LHCoptim(n,vars,100)
    scaled_plan = scaleLHC(plan,repeat([(-0.1,0.1)], vars))
    for i in 1:n
        if style isa initRow
            initializations[i] = scaled_plan[[i],:] 
        elseif style isa initMatrix
            initializations[i] = reshape(scaled_plan[[i],:], Int64(nrows), Int64(vars/nrows))
        end
    end
    return initializations
end


"""
Manifold optimization for orthogonality constraint
The manifold Orth() is used to optimize functions over N x n matrices with orthogonal rows, i.e. such that X'X is diagonal.
Orth() uses an SVD algorithm to compute the retraction.
"""
struct Orth <: Manifold
end

function Optim.retract!(M::Orth, X::Matrix{Float64})
    U,S,V = svd(X)
    X .= S.*U*V'
end

Optim.project_tangent!(M::Orth, G, X) = (XG = X'G; G .-= X*((XG .+ XG')./2))


"""
    algSNCA(x_matrix, y; objective)

Run the SNCA algorithm with data x_matrix and y, with defined objective scaling. Returns the optimal objective value and solution.

# Arguments:

- `x_matrix` should be a Matrix, each column of which is a feature vector
- `y` should be an `AbstractVector` containing the classes of the points represented by the columns of `x_matrix`.
- `objective` is a named argument which whould be of type `NCAMethod`. Choose from:
    - `NCAStandard()` for standard (which is the default).
    - `NCALog()` for log.
```

"""

function algSNCA(x_matrix, y::AbstractVector; objective=NCAStandard(), inits=10)
    size(x_matrix,2)==length(y) || throw(ArgumentError("number of columns of x_matrix and length of y should be the same."))
    D = size(x_matrix, 1)
    x = svectorscopy(x_matrix, Val(D))
    A_res = Array{Float64}(undef, 0, D)
    if inits == 1
        LHSinitializations = [rand(1, D)]
    else
        LHSinitializations = initsLHS(D, n=inits)
    end
    objvalues = Vector{Float64}(undef, length(LHSinitializations))
    solns = Array{Array{Float64, 2}, 1}(undef, length(LHSinitializations))
    aggP = Matrix(I, D, D)
    for j in 1:D
        @showprogress "row $j: " for i in 1:length(LHSinitializations) 
            Random.seed!(i*j)
            resSNCA = optimize(A -> SNCAincremental(A, vcat(zeros(1,D), A_res), x, aggP, y, objective = objective, dims = Val(1)), LHSinitializations[i],
                LBFGS(linesearch=LineSearches.BackTracking()))
            objvalues[i] = resSNCA.minimum
            solns[i] = resSNCA.minimizer
        end
        rankedsolns = sortperm(objvalues)
        best = rankedsolns[1]
        aⱼ = solns[best]
        A_res = vcat(A_res, aⱼ)
        if j > 1
            oldobj = SNCA(A_res[1:(end-1),:], x, y, objective=objective, dims=Val(j-1))
            newobj = SNCA(A_res, x, y, objective=objective, dims=Val(j))
            if round(oldobj, digits = 5) <= round(newobj, digits = 5)
                A_res = A_res[1:(end-1),:]
                break                
            end
        end
        if j < D
            aⱼhat = aⱼ/norm(aⱼ)
            P = Matrix(I, D, D) - transpose(aⱼhat) * aⱼhat
            aggP = aggP * P
            LHSinitializations = [Matrix(transpose(P*LHSinitializations[i]')) for i in 1:length(LHSinitializations)]
        end
    end
    SolutionD = size(A_res, 1)
    A_res_final = optimize(A -> SNCA(A, x, y, objective=objective, dims = Val(SolutionD)), A_res, Optim.LBFGS(manifold=Orth(), linesearch=LineSearches.BackTracking())).minimizer
    objvalue = SNCA(A_res_final, x, y, objective=objective, dims=Val(SolutionD))
    return objvalue, A_res_final
end

