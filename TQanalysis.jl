include("src/functionswrapper.jl")
TQ_data = load_object("dat/TQ_data.jld2")

x_matrix = Matrix{Float64}(transpose(TQ_data[:,1:(end-1)]));
x = svectorscopy(x_matrix, Val(4));
y = Vector{Int64}(TQ_data[:,end]);

##learn SNCA matrix for Figure 2
Random.seed!(1)
objvalue_SNCA, A_SNCA = algSNCA(x_matrix, y, objective=NCALog())
save_object("res/TQ_SNCA_A.jld2", A_SNCA)

#for Figure 2 (bottom)
zerosklkth_SNCA = StateKLkth([0,0,0,0], x_matrix, y, A_SNCA);
save_object("res/TQ_SNCA_klkth_0.jld2", zerosklkth_SNCA)


## learn 2d NCA matrix for Figure 3
Random.seed!(1)
initializations = initsLHS(8, n=10, style=initMatrix(), nrows=2);
objvalues = Vector{Float64}(undef, length(initializations));
solns = Array{Array{Float64, 2}, 1}(undef, length(initializations));
for j in 1:length(initializations)
    res = optimize(Optim.only_fg!((F,G,A) -> NCArepeatsfg!(F,G,A,x,y,objective=NCALog(),dims=Val(2))), initializations[j], LBFGS(linesearch=LineSearches.BackTracking()))
    objvalues[j] = res.minimum
    solns[j] = res.minimizer
end
rankedsolns = sortperm(objvalues);
best = rankedsolns[1];
A_NCA = solns[best]
save_object("res/TQ_NCA_A.jld2", A_NCA)

#for Figure 3 (bottom)
zerosklkth_NCA = StateKLkth([0,0,0,0], x_matrix, y, A_NCA);
save_object("res/TQ_NCA_klkth_0.jld2", zerosklkth_NCA)


##for Figure 4
SNCAKLkth = Vector{Float64}(undef, 35);
for k in 1:35 
    SNCAKLkth[k] = knnMSE(x_matrix, y, k, A_SNCA)[2]
end
NCAKLkth = Vector{Float64}(undef, 35);
for k in 1:35 
    NCAKLkth[k] = knnMSE(x_matrix, y, k, A_NCA)[2]
end
save_object("res/TQ_SNCA_klkth.jld2", SNCAKLkth)
save_object("res/TQ_NCA_klkth.jld2", NCAKLkth)


##for Figure 5 (left) and Figure 6
NCA_Acc = Array{Matrix{Float64}, 1}(undef, 10);
SNCA_Acc = Array{Matrix{Float64}, 1}(undef, 10);
Euclidean_Acc = Array{Matrix{Float64}, 1}(undef, 10);
SNCA_time = Array{Vector{Float64}, 1}(undef, 10);
NCA_time_pointwise = Array{Vector{Float64}, 1}(undef, 10);
for n in 1:10
    println("n =  $n 000:")
    x_n = x_matrix[:,1:(10000*n)]
    y_n = y[1:(10000*n)]
    Xs, ys = sliceData(x_n, y_n, n*1000)
    NCA_Acc[n] = NCAiterate(Xs, ys, objective=NCALog())[1]
    SNCA_Acc[n], SNCA_time[n] = SNCArowiterate(Xs, ys, objective=NCALog())
    NCA_time_pointwise[n] = NCAiterate(Xs, ys, objective=NCALog(), version="pointwise")[2]
    Euclidean_Acc[n] = Euclideanres(x_matrix, y, 10)
end
save_object("res/TQ_NCA_Acc.jld2", NCA_Acc)
save_object("res/TQ_SNCA_Acc.jld2", SNCA_Acc)
save_object("res/TQ_Euclidean_Acc.jld2", Euclidean_Acc)
save_object("res/TQ_NCA_time_pointwise.jld2", NCA_time_pointwise)
save_object("res/TQ_SNCA_time.jld2", SNCA_time)

#for Figure 5 (right)
NCA_Acc_test = Array{Matrix{Float64}, 1}(undef, 10);
SNCA_Acc_test = Array{Matrix{Float64}, 1}(undef, 10);
Euclidean_Acc_test = Array{Matrix{Float64}, 1}(undef, 10);
NNopt_Acc_test = Array{Matrix{Float64}, 1}(undef, 10);
for n in 1:10
    println("n = $n 000")
    x_n = x_matrix[:,1:(10000*n)]
    y_n = y[1:(10000*n)]
    Xs, ys = sliceData(x_n, y_n, n*1000)
    NCA_Acc_test[n] = NCAiterateunseen(Xs, ys, x_matrix, y, 1, objective=NCALog())
    SNCA_Acc_test[n] = SNCArowiterateunseen(Xs, ys, x_matrix, y, 1, objective=NCALog())
    Euclidean_Acc_test[n] = Euclideaniterateunseen(Xs, ys, x_matrix, y, 1)
    NNopt_Acc_test[n] = NNoptunseen(Xs, ys, x_matrix, y, 1)
end

save_object("res/TQ_NCA_Acc_test.jld2", NCA_Acc_test)
save_object("res/TQ_SNCA_Acc_test.jld2", SNCA_Acc_test)
save_object("res/TQ_Euclidean_Acc_test.jld2", Euclidean_Acc_test)
save_object("res/TQ_NNopt_Acc_test.jld2", NNopt_Acc_test)