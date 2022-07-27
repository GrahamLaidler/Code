function sliceData(X::AbstractMatrix{T}, y::AbstractVector, n) where T
    #X has d rows, and is to be split into segments of n columns
    D, ncol = size(X)
    ncol == length(y) || throw(ArgumentError("X and y should be the same length."))
    parts = Int64(ncol/n)
    X_partition = Array{Array{T, 2}, 1}(undef, parts)
    y_partition = Vector{Int64}[Vector{Int64}(undef, n) for _ in 1:parts]

    for i = 1:parts
        X_partition[i] = X[:, (n*(i-1) + 1):(n*i)]
        y_partition[i] = y[(n*(i-1) + 1):(n*i)]
    end
    return X_partition, y_partition
end

function balanced_kfold(y::Vector, k::Integer)
    d = Dict{eltype(y), Vector{Int}}()
    folds = [Int[] for i in 1:k]
    for (i, v) in enumerate(y)
        push!(get!(d, v, Int[]), i)
    end
    for val in values(d)
        shuffle!(val)
        for (i, pos) in enumerate(val)
            push!(folds[mod1(i, k)], pos)
        end
    end
    folds
end

function knnMSE(x_matrix, y, k, A)
    D = size(x_matrix, 1)
    x = svectorscopy(x_matrix, Val(D))
    cellsx = countmap(x)
    k < length(cellsx) || throw(ArgumentError("k is greater than the number of neighbor sets."))
    x1 = svectorscopy(x_matrix[:,y.==1], Val(D))
    cellsx1 = countmap(x1)
    M = transpose(A) * A
    d = SqMahalanobis(M, skipchecks = true)
    distances = Vector{Float64}(undef, length(cellsx))
    knncorrect = 0
    TKLkth = 0.0
    for xᵢ in keys(cellsx)
        if haskey(cellsx1, xᵢ)
            emp1xᵢ = cellsx1[xᵢ]/cellsx[xᵢ]
        else 
            emp1xᵢ = 0.0
        end
        for (j,xⱼ) ∈ enumerate(keys(cellsx))
            distances[j] = d(xᵢ, xⱼ) + (Inf * (xᵢ == xⱼ)) 
        end
        order = sortperm(distances) 
        neighbours = order[1:k]
        kth = order[k:k]
        for i in 1:length(distances) 
            if distances[order[i]] == distances[kth] && i != k
                append!(kth, order[i])
            end
        end
        tieindx = k+1
        while distances[order[tieindx]] == distances[order[k]] && tieindx <= length(distances)-1
            append!(neighbours, order[tieindx])
            tieindx += 1
        end
        count = 0    
        count1 = 0
        for i in neighbours
            count += cellsx[collect(keys(cellsx))[i]]
            if haskey(cellsx1, collect(keys(cellsx))[i])
                count1 += cellsx1[collect(keys(cellsx))[i]]
            end
        end
        emp1neighbours = count1/count  
        countkth = 0    
        count1kth = 0
        for i in kth
            countkth += cellsx[collect(keys(cellsx))[i]]
            if haskey(cellsx1, collect(keys(cellsx))[i])
                count1kth += cellsx1[collect(keys(cellsx))[i]]
            end
        end
        emp1kth = count1kth/countkth   
        if haskey(cellsx1, xᵢ)
            knncorrect += cellsx1[xᵢ]*round(emp1neighbours) + (cellsx[xᵢ]-cellsx1[xᵢ])*(1-round(emp1neighbours))
        else
            knncorrect += cellsx[xᵢ]*(1-round(emp1neighbours))
        end
        kth_dist = [emp1kth, 1-emp1kth]
        empirical_dist = [emp1xᵢ, 1-emp1xᵢ]
        klkth_xᵢ = kldivergence(empirical_dist, kth_dist)
        TKLkth += cellsx[xᵢ] * klkth_xᵢ
    end
    knnacc = knncorrect/sum(values(cellsx))
    KLkth = TKLkth/sum(values(cellsx))
    return knnacc, KLkth
end


function NCAiterate(Xs, ys; objective=NCAStandard())
    D = size(Xs[1],1)
    maxk = maximum(length(countmap(svectorscopy(Xs[i], Val(D))))-1 for i in 1:length(Xs))
    knnaccMat = Matrix{Float64}(undef, maxk, length(Xs))
    @showprogress "NCA 10 iterations: " for i in 1:length(Xs) 
        Random.seed!(i)
        initializations = initsLHS(D^2, n=10, style=initMatrix(), nrows=4)
        x = svectorscopy(Xs[i], Val(D))
        objvalues = Vector{Float64}(undef, length(initializations))
        solns = Array{Array{Float64, 2}, 1}(undef, length(initializations))
        for j in 1:length(initializations)
            r = size(initializations[j], 1)
            Random.seed!(j)
            res = optimize(Optim.only_fg!((F,G,A) -> NCArepeatsfg!(F,G,A,x,ys[i],objective=objective,dims=Val(r))), initializations[j], LBFGS(linesearch=LineSearches.BackTracking()))
            objvalues[j] = res.minimum
            solns[j] = res.minimizer
        end
        rankedsolns = sortperm(objvalues)
        best = rankedsolns[1]
        for k in 1:(length(countmap(x))-1)
            knnacc = knnMSE(x_matrix, y, k, solns[best])[1]
            knnaccMat[k,i] = knnacc
        end
    end
    return knnaccMat
end

function SNCArowiterate(Xs, ys; objective=NCAStandard())
    D = size(Xs[1],1)
    maxk = maximum(length(countmap(svectorscopy(Xs[i], Val(D))))-1 for i in 1:length(Xs))
    knnaccMat = Matrix{Float64}(undef, maxk, length(Xs))
    @showprogress "SNCA 10 iterations: " for i in 1:length(Xs) 
        x = svectorscopy(Xs[i], Val(D))
        objvalue, A_res = algSNCA(Xs[i], ys[i], objective=objective)
        for k in 1:(length(countmap(x))-1)
            knnacc = knnMSE(x_matrix, y, k, A_res)[1]
            knnaccMat[k,i] = knnacc
        end
    end
    return knnaccMat
end

function Euclideanres(x_matrix, y, iterations)
    D = size(x_matrix,1)
    maxk = length(countmap(svectorscopy(x_matrix, Val(D))))-1
    knnaccMat = Matrix{Float64}(undef, maxk, iterations)
    for k in 1:maxk 
        knnacc = knnMSE(x_matrix, y, k, Matrix(I, D, D))[1]
        for i in 1:iterations
            knnaccMat[k,i] = knnacc
        end
    end
    return knnaccMat
end


function NCAiterateunseen(Xs, ys, xlarge_matrix, ylarge, rndm; objective=NCAStandard())
    D = size(Xs[1],1)
    xfull = svectorscopy(xlarge_matrix, Val(D))
    knnaccMat = zeros(6, length(Xs))
    initializations = initsLHS(D^2, n=10, style=initMatrix(), nrows=4)
    Random.seed!(rndm)
    Statesorder = shuffle(unique(xfull))
    for i in 1:length(Xs) 
        x = svectorscopy(Xs[i], Val(D))
        y = ys[i]
        @showprogress "NCA unseen dataset $i: " for l in 1:6
            metric_trainx = x[[x[n] in Statesorder[(6*(l-1)+1):(6*l)] for n in 1:length(x)].==false]   #
            metric_trainy = y[[x[n] in Statesorder[(6*(l-1)+1):(6*l)] for n in 1:length(x)].==false]   #
            indx = [xfull[n] in Statesorder[(6*(l-1)+1):(6*l)] for n in 1:length(xfull)]
            kNN_trainx = xlarge_matrix[:,indx.==false]
            kNN_trainy = ylarge[indx.==false]
            kNN_testx = xlarge_matrix[:,indx.==true]
            kNN_testy = ylarge[indx.==true]
            objvalues = Vector{Float64}(undef, length(initializations))
            solns = Array{Array{Float64, 2}, 1}(undef, length(initializations))
            for j in 1:length(initializations)
                r = size(initializations[j], 1)
                Random.seed!(i*j)
                res = optimize(Optim.only_fg!((F,G,A) -> NCArepeatsfg!(F,G,A,metric_trainx,metric_trainy,objective=objective,dims=Val(r))), initializations[j], LBFGS(linesearch=LineSearches.BackTracking()))
                objvalues[j] = res.minimum
                solns[j] = res.minimizer
            end
            rankedsolns = sortperm(objvalues)
            best = rankedsolns[1]
            knncorrect = knnMSEtest(kNN_trainx, kNN_trainy, kNN_testx, kNN_testy, 1, solns[best])  
            knnaccMat[l,i] += knncorrect/size(kNN_testx, 2)
        end
    end
    return knnaccMat 
end

function SNCArowiterateunseen(Xs, ys, xlarge_matrix, ylarge, rndm; objective=NCAStandard())
    D = size(Xs[1],1)
    xfull = svectorscopy(xlarge_matrix, Val(D))
    knnaccMat = zeros(6, length(Xs)) 
    Random.seed!(rndm)
    Statesorder = shuffle(unique(xfull))
    for i in 1:length(Xs) 
        x = svectorscopy(Xs[i], Val(D))
        y = ys[i]
        @showprogress "SNCA unseen dataset $i: " for l in 1:6    #
            metric_trainx = Xs[i][:,[x[n] in Statesorder[(6*(l-1)+1):(6*l)] for n in 1:length(x)].==false]   
            metric_trainy = y[[x[n] in Statesorder[(6*(l-1)+1):(6*l)] for n in 1:length(x)].==false]   #
            indx = [xfull[n] in Statesorder[(6*(l-1)+1):(6*l)] for n in 1:length(xfull)]
            kNN_trainx = xlarge_matrix[:,indx.==false]
            kNN_trainy = ylarge[indx.==false]
            kNN_testx = xlarge_matrix[:,indx.==true]
            kNN_testy = ylarge[indx.==true]
            objvalues = Vector{Float64}(undef, 10)
            solns = Array{Array{Float64, 2}, 1}(undef, 10)
            objvalue, A_res = algSNCA(metric_trainx, metric_trainy, objective=objective)
            knncorrect = knnMSEtest(kNN_trainx, kNN_trainy, kNN_testx, kNN_testy, 1, A_res)
            knnaccMat[l,i] += knncorrect/size(kNN_testx, 2)
        end
    end
    return knnaccMat
end

function Euclideaniterateunseen(Xs, ys, xlarge_matrix, ylarge, rndm)
    D = size(Xs[1],1)
    xfull = svectorscopy(xlarge_matrix, Val(D)) 
    knnaccMat = zeros(6, length(Xs))
    Random.seed!(rndm)
    Statesorder = shuffle(unique(xfull))
    for i in 1:length(Xs)
        x = svectorscopy(Xs[i], Val(D))
        y = ys[i]
        @showprogress "Euclidean unseen dataset $i: " for l in 1:6  
            indx = [xfull[n] in Statesorder[(6*(l-1)+1):(6*l)] for n in 1:length(xfull)]
            kNN_trainx = xlarge_matrix[:,indx.==false]
            kNN_trainy = ylarge[indx.==false]
            kNN_testx = xlarge_matrix[:,indx.==true]
            kNN_testy = ylarge[indx.==true]
            knncorrect = knnMSEtest(kNN_trainx, kNN_trainy, kNN_testx, kNN_testy, 1, Matrix{Float64}(I, D, D))
            knnaccMat[l,i] += knncorrect/size(kNN_testx, 2)
        end
    end
    return knnaccMat 
end


function knnMSEtest(x_matrix, y, xtest, ytest, k, A)
    D = size(x_matrix, 1)
    x = svectorscopy(x_matrix, Val(D))
    cellsxtrain = countmap(x)
    k <= length(cellsxtrain) || throw(ArgumentError("k is greater than the number of neighbor sets."))
    x1 = svectorscopy(x_matrix[:,y.==1], Val(D))
    cellsx1train = countmap(x1)
    cellsxtest = countmap(svectorscopy(xtest, Val(D)))
    x1test = svectorscopy(xtest[:,ytest.==1], Val(D))
    cellsx1test = countmap(x1test)
    d = SqEuclidean()
    distances = Vector{Float64}(undef, length(cellsxtrain))
    knncorrect = 0
    for xᵢ in keys(cellsxtest)
        if haskey(cellsx1test, xᵢ)
            emp1xᵢ = cellsx1test[xᵢ]/cellsxtest[xᵢ]
        else 
            emp1xᵢ = 0.0
        end
        for (j,xⱼ) ∈ enumerate(keys(cellsxtrain))
            distances[j] = d(A*xᵢ, A*xⱼ) + (Inf * (xᵢ == xⱼ)) 
        end
        order = sortperm(distances) 
        neighbours = order[1:k]
        kth = order[k:k]
        for i in 1:length(distances) 
            if distances[order[i]] == distances[kth] && i != k
                append!(kth, order[i])
            end
        end
        tieindx = k+1
        while distances[order[tieindx]] == distances[order[k]] && tieindx <= length(distances)-1
            append!(neighbours, order[tieindx])
            tieindx += 1
        end
        count = 0 
        count1 = 0
        for i in neighbours
            count += cellsxtrain[collect(keys(cellsxtrain))[i]]
            if haskey(cellsx1train, collect(keys(cellsxtrain))[i])
                count1 += cellsx1train[collect(keys(cellsxtrain))[i]]
            end
        end
        emp1neighbours = count1/count 
        if haskey(cellsx1test, xᵢ)
            knncorrect += cellsx1test[xᵢ]*round(emp1neighbours) + (cellsxtest[xᵢ]-cellsx1test[xᵢ])*(1-round(emp1neighbours))
        else
            knncorrect += cellsxtest[xᵢ]*(1-round(emp1neighbours))
        end
    end
    return knncorrect
end

function StateKLkth(statevec, x_matrix, y, A)
    #find the KL to kth values from a particular state
    #x_matrix and y should be large n
    #statevec is a Vector
    D = size(x_matrix, 1)
    x = svectorscopy(x_matrix, Val(D))
    cellsx = countmap(x)
    state = SVector{D, Float64}(statevec)
    haskey(cellsx, state) || throw(ArgumentError("state not contained in x_matrix"))
    x1 = svectorscopy(x_matrix[:,y.==1], Val(D))
    cellsx1 = countmap(x1)
    M = transpose(A) * A
    d = SqMahalanobis(M, skipchecks = true)
    distances = Vector{Float64}(undef, length(cellsx))
    kldivs = Vector{Float64}(undef, length(cellsx)-1)
    emp1s = Vector{Float64}(undef, length(cellsx)-1)
    densities = Vector{Float64}(undef, length(cellsx)-1)
    if haskey(cellsx1, state)
        emp1 = cellsx1[state]/cellsx[state]
    else 
        emp1 = 0.0
    end
    for (j,xⱼ) ∈ enumerate(keys(cellsx))
        distances[j] = d(state, xⱼ) + (Inf * (state == xⱼ)) 
    end
    order = sortperm(distances) 
        for k in 1:(length(cellsx)-1)
            kth = order[k:k]
            for i in 1:length(distances) 
                if distances[order[i]] == distances[kth] && i != k
                    append!(kth, order[i])
                end
            end
            countkth = 0   
            count1kth = 0
            for i in kth
                countkth += cellsx[collect(keys(cellsx))[i]]
                if haskey(cellsx1, collect(keys(cellsx))[i])
                    count1kth += cellsx1[collect(keys(cellsx))[i]]
                end
            end
            emp1kth = count1kth/countkth  
            kth_dist = [emp1kth, 1-emp1kth]
            empirical_dist = [emp1, 1-emp1]
            klkth = kldivergence(empirical_dist, kth_dist)
            kldivs[k] = klkth
            emp1s[k] = emp1kth
            densities[k] = countkth
    end
    return kldivs, emp1s, densities
end

function NNopt(x_matrix, y)
    D = size(x_matrix, 1)
    x = svectorscopy(x_matrix, Val(D))
    cellsx = countmap(x)
    x1 = svectorscopy(x_matrix[:,y.==1], Val(D))
    cellsx1 = countmap(x1)
    knncorrect = 0
    emps = Vector{Float64}(undef, length(cellsx))
    for (i,xᵢ) in enumerate(keys(cellsx))
        if haskey(cellsx1, xᵢ)
            emp1xᵢ = cellsx1[xᵢ]/cellsx[xᵢ]
        else 
            emp1xᵢ = 0.0
        end
        emps[i] = emp1xᵢ
    end
    for (i,xᵢ) in enumerate(keys(cellsx)) 
        kldivs = Vector{Float64}(undef, length(cellsx))
        for (j,xⱼ) in enumerate(keys(cellsx))
            kldivs[j] = kldivergence([emps[i], 1-emps[i]], [emps[j], 1-emps[j]])
        end
        kldivs[i] = Inf
        order = sortperm(kldivs)
        closest = order[1]
        if haskey(cellsx1, xᵢ)
            knncorrect += cellsx1[xᵢ]*round(emps[closest]) + (cellsx[xᵢ]-cellsx1[xᵢ])*(1-round(emps[closest]))
        else
            knncorrect += cellsx[xᵢ]*(1-round(emps[closest]))
        end
    end
    NNoptacc = knncorrect/sum(values(cellsx))
    return NNoptacc
end

function NNoptunseen(Xs, ys, xlarge_matrix, ylarge, rndm)
    D = size(Xs[1],1)
    xfull = svectorscopy(xlarge_matrix, Val(D))
    cellsx = countmap(xfull)
    x1 = svectorscopy(xlarge_matrix[:,ylarge.==1], Val(D))
    cellsx1 = countmap(x1)
    NNoptaccMat = zeros(6, length(Xs))
    Random.seed!(rndm)
    Statesorder = shuffle(unique(xfull))
    for i in 1:length(Xs)
        x = svectorscopy(Xs[i], Val(D))
        y = ys[i]
        for l in 1:6    
            indx = [xfull[n] in Statesorder[(6*(l-1)+1):(6*l)] for n in 1:length(xfull)]
            kNN_trainx = xlarge_matrix[:,indx.==false]
            kNN_trainy = ylarge[indx.==false]
            kNN_testx = xlarge_matrix[:,indx.==true]
            kNN_testy = ylarge[indx.==true]
            xtrain = svectorscopy(kNN_trainx, Val(D))
            cellsxtrain = countmap(xtrain)
            x1train = svectorscopy(kNN_trainx[:,kNN_trainy.==1], Val(D))
            cellsx1train = countmap(x1train)
            xtest = svectorscopy(kNN_testx, Val(D))
            cellsxtest = countmap(xtest)
            x1test = svectorscopy(kNN_testx[:,kNN_testy.==1], Val(D))
            cellsx1test = countmap(x1test)
            NNoptcorrect = 0
            empstest = Vector{Float64}(undef, length(cellsxtest))
            empstrain = Vector{Float64}(undef, length(cellsxtrain))
            for (i,xᵢ) in enumerate(keys(cellsxtest))
                if haskey(cellsx1test, xᵢ)
                    emp1xᵢ = cellsx1test[xᵢ]/cellsxtest[xᵢ]
                else 
                    emp1xᵢ = 0.0
                end
                empstest[i] = emp1xᵢ
            end
            for (i,xᵢ) in enumerate(keys(cellsxtrain))
                if haskey(cellsx1train, xᵢ)
                    emp1xᵢ = cellsx1train[xᵢ]/cellsxtrain[xᵢ]
                else 
                    emp1xᵢ = 0.0
                end
                empstrain[i] = emp1xᵢ
            end
            for (i,xᵢ) in enumerate(keys(cellsxtest)) 
                kldivs = Vector{Float64}(undef, length(cellsxtrain))
                for (j,xⱼ) in enumerate(keys(cellsxtrain))
                    kldivs[j] = kldivergence([empstest[i], 1-empstest[i]], [empstrain[j], 1-empstrain[j]])
                end
                order = sortperm(kldivs)
                closest = order[1]
                if haskey(cellsx1test, xᵢ)
                    NNoptcorrect += cellsx1test[xᵢ]*round(empstrain[closest]) + (cellsxtest[xᵢ]-cellsx1test[xᵢ])*(1-round(empstrain[closest]))
                else
                    NNoptcorrect += cellsxtest[xᵢ]*(1-round(empstrain[closest]))
                end
            end
            NNoptaccMat[l,i] += NNoptcorrect/size(kNN_testx, 2)
        end
    end
    return NNoptaccMat
end

function knnSetstesttrain(x_test, y_test, x_train, y_train, k, A)
    D = size(x_train, 1)
    cellsxtrain = countmap(svectorscopy(x_train, Val(D)))
    k <= length(cellsxtrain) || throw(ArgumentError("k is greater than the number of neighbor sets."))
    xtrain1 = svectorscopy(x_train[:,y_train.==1], Val(D))
    cellsxtrain1 = countmap(xtrain1)
    cellsxtest = countmap(svectorscopy(x_test, Val(D)))
    xtest1 = svectorscopy(x_test[:,y_test.==1], Val(D))
    cellsxtest1 = countmap(xtest1)
    d = SqEuclidean()
    distances = Vector{Float64}(undef, length(cellsxtrain))
    knncorrect = 0
    for xᵢ in keys(cellsxtest)
        if haskey(cellsxtest1, xᵢ)
            emp1xᵢ = cellsxtest1[xᵢ]/cellsxtest[xᵢ]
        else 
            emp1xᵢ = 0.0
        end
        for (j,xⱼ) ∈ enumerate(keys(cellsxtrain))
            distances[j] = d(A*xᵢ, A*xⱼ) + (Inf * (xᵢ == xⱼ)) 
        end
        order = sortperm(distances) 
        neighbours = order[1:k]
        ties = k+1
        while distances[order[ties]] == distances[order[k]] && ties <= length(distances)-1
            append!(neighbours, order[ties])
            ties += 1
        end
        count = 0  
        count1 = 0
        for i in neighbours
            count += cellsxtrain[collect(keys(cellsxtrain))[i]]
            if haskey(cellsxtrain1, collect(keys(cellsxtrain))[i])
                count1 += cellsxtrain1[collect(keys(cellsxtrain))[i]]
            end
        end
        emp1neighbours = count1/count  
        if haskey(cellsxtest1, xᵢ)
            knncorrect += cellsxtest1[xᵢ]*round(emp1neighbours) + (cellsxtest[xᵢ]-cellsxtest1[xᵢ])*(1-round(emp1neighbours))
        else
            knncorrect += cellsxtest[xᵢ]*(1-round(emp1neighbours))
        end
    end
    knnacc = knncorrect/sum(values(cellsxtest))
    return knnacc
end