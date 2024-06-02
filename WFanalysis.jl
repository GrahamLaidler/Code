include("src/functionswrapper.jl")
WF_data = load_object("dat/WF_data.jld2")

x_matrix = Matrix{Float64}(transpose(WF_data[:,1:(end-1)]));
x = svectorscopy(x_matrix, Val(11));
y = Vector{Int64}(WF_data[:,end]);

I₁₁ = Matrix(I,11,11)
SNCA_Metrics = Array{Matrix{Float64}}(undef, 10);
#NCA_Metrics = Array{Matrix{Float64}}(undef, 10);
SNCA_kNN = Matrix{Float64}(undef, 30, 10);
NCA_kNN = Matrix{Float64}(undef, 30, 10);
Euclidean_kNN = Matrix{Float64}(undef, 30, 10);

for i in 1:5

    Random.seed!(1234*i)
    x_partitions = balanced_kfold(y, 2)
    for m in 1:2
        println("iteration $i Dataset $m")
        x_matrixm = x_matrix[:,x_partitions[m]]
        xm = svectorscopy(x_matrixm, Val(11))
        ym = y[x_partitions[m]]
        rare = findall(x -> x < 25, countmap(xm))
        rare_indx = findall(in(rare), xm)
        x_metrictrain = x_matrixm[:, setdiff(1:size(x_matrixm, 2), rare_indx)]
        x_metrictrainSvec = svectorscopy(x_metrictrain, Val(11))
        y_metrictrain = ym[setdiff(1:size(x_matrixm, 2), rare_indx)]
        x_kNN_rare = x_matrixm[:,rare_indx]
        y_kNN_rare = ym[rare_indx]

        Random.seed!(2*(i-1) + m)
        objvalue_SNCA, A_SNCA = algSNCA(x_metrictrain, y_metrictrain, objective=NCALog())
        SNCA_Metrics[2*(i-1) + m] = A_SNCA
        Random.seed!(2*(i-1) + m)
        initializations = initsLHS(11^2, n=10, style=initMatrix(), nrows=11);
        objvalues = Vector{Float64}(undef, length(initializations));
        solns = Array{Array{Float64, 2}, 1}(undef, length(initializations));
        @showprogress "progress NCA:" for j in 1:length(initializations)
            r = size(initializations[j], 1)
            Random.seed!(m*j)
            res = optimize(Optim.only_fg!((F,G,A) -> NCArepeatsfg!(F,G,A,x_metrictrainSvec,y_metrictrain,objective=NCALog(),dims=Val(r))), initializations[j],
                LBFGS(linesearch=LineSearches.BackTracking()))
            objvalues[j] = res.minimum
            solns[j] = res.minimizer
        end
        rankedsolns = sortperm(objvalues)
        best = rankedsolns[1]
        A_NCA = solns[best]
        #NCA_Metrics[2*(i-1) + m] = A_NCA
        @showprogress "progress kNN sets: " for k in 1:30
            SNCA_kNN[k,2*(i-1) + m] = knnSetstesttrain(x_kNN_rare, y_kNN_rare, x_metrictrain, y_metrictrain, k, A_SNCA)
            NCA_kNN[k,2*(i-1) + m] = knnSetstesttrain(x_kNN_rare, y_kNN_rare, x_metrictrain, y_metrictrain, k, A_NCA)
            Euclidean_kNN[k,2*(i-1) + m] = knnSetstesttrain(x_kNN_rare, y_kNN_rare, x_metrictrain, y_metrictrain, k, I₁₁)
        end
    end
end
save_object("res/WF_SNCA_kNN.jld2", SNCA_kNN)
save_object("res/WF_NCA_kNN.jld2", NCA_kNN)
save_object("res/WF_Euclidean_kNN.jld2", Euclidean_kNN)

#save example SNCA solution matrix for Figure 7 (left)
save_object("res/WF_SNCA_A.jld2", SNCA_Metrics[1])