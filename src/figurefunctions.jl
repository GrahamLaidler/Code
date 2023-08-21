function plot2d(x_matrix, y, A)
    #prepare to plot data projected in the first two rows of A
    D = size(x_matrix,1)
    x =svectorscopy(x_matrix, Val(D))
    cellsx=sort(countmap(x); byvalue=true, rev=true)
    x1 = svectorscopy(x_matrix[:,y.==1], Val(D))
    cellsx1 = countmap(x1)
    emp1 = Vector{Float64}(undef, length(cellsx))
    for (l, bₗ) in enumerate(keys(cellsx)) 
        if haskey(cellsx1, bₗ)
            emp1[l] = cellsx1[bₗ]/cellsx[bₗ]
        else
            emp1[l] = 0.0
        end
    end
    densities = [cellsx[bₗ] for bₗ in keys(cellsx)]
    points = [bₗ for bₗ in keys(cellsx)]
    trans_points = [A*bₗ for bₗ in points]
    return trans_points, densities, emp1
end 
    

function Figure2()
    TQ_SNCA_A = load_object("res/TQ_SNCA_A.jld2")
    TQ_data = load_object("dat/TQ_data.jld2")
    x_matrix = Matrix{Float64}(transpose(TQ_data[:,1:(end-1)]));
    y = Vector{Int64}(TQ_data[:,end]);
    trans_points, densities, emp1 = plot2d(x_matrix, y, TQ_SNCA_A)

    SNCAscatter = scatter([-bₗ[1] for bₗ in trans_points], [bₗ[2] for bₗ in trans_points],
        markersize = densities./500,
        markerz = emp1,
        markerstrokewidth = 0.5,
        legend = false,
        ylim = (-1.8, 3),
        colorbar = false,
        dpi=600,
        size = (600,150))

    zerosklkth_SNCA = load_object("res/TQ_SNCA_klkth_0.jld2")
    SNCAKL = plot(zerosklkth_SNCA[1], linecolor = "black", alpha = 0.5, linewidth = 2,
        xlabel = L"k", ylabel = "KL divergence", size = (600,150), dpi=600)
    SNCAKL = scatter!(zerosklkth_SNCA[1],
        markersize = zerosklkth_SNCA[3]./500,
        markerz = zerosklkth_SNCA[2],
        markerstrokewidth = 0.5,
        legend = false,
        ylim=(-0.05,0.8),
        colorbar = false,
        dpi=600)
    h2 = scatter([0,0], [0,1], zcolor=[minimum(zerosklkth_SNCA[2]),maximum(zerosklkth_SNCA[2])],
        xlims=(1,1.1), framestyle=:none, label="", colorbar_title=L"\hat{q}(Y=1|X)", grid=false)
    l = @layout [grid(2, 1) a{0.035w}]
    plots_SNCA = plot(SNCAscatter, SNCAKL, h2, layout=l)
    return plots_SNCA
end

function Figure3()
    TQ_NCA_A = load_object("res/TQ_NCA_A.jld2")
    TQ_data = load_object("dat/TQ_data.jld2")
    x_matrix = Matrix{Float64}(transpose(TQ_data[:,1:(end-1)]));
    y = Vector{Int64}(TQ_data[:,end]);
    trans_points, densities, emp1 = plot2d(x_matrix, y, TQ_NCA_A)

    NCAscatter = scatter([bₗ[1] for bₗ in trans_points], [bₗ[2] for bₗ in trans_points],
        markersize = densities./500,
        markerz = emp1,
        markerstrokewidth = 0.5,
        legend = false,
        ylim = (-66,12),
        colorbar = false,
        dpi=600,
        size = (600,150))

    zerosklkth_NCA = load_object("res/TQ_NCA_klkth_0.jld2")
    NCAKL = plot(zerosklkth_NCA[1], linecolor = "black", alpha = 0.5, linewidth = 2,
        xlabel = L"k", ylabel = "KL divergence", size = (600,150), dpi=600)
    NCAKL = scatter!(zerosklkth_NCA[1],
        markersize = zerosklkth_NCA[3]./500,
        markerz = zerosklkth_NCA[2],
        markerstrokewidth = 0.5,
        legend = false,
        ylim=(-0.05,0.8),
        colorbar = false,
        dpi=600)
    h2 = scatter([0,0], [0,1], zcolor=[minimum(zerosklkth_NCA[2]),maximum(zerosklkth_NCA[2])],
        xlims=(1,1.1), framestyle=:none, label="", colorbar_title=L"\hat{q}(Y=1|X)", grid=false)
    l = @layout [grid(2, 1) a{0.035w}]
    plots_NCA = plot(NCAscatter, NCAKL, h2, layout=l)
    return plots_NCA
end

function Figure4()
    SNCAKLkth = load_object("res/TQ_SNCA_klkth.jld2")
    NCAKLkth = load_object("res/TQ_NCA_klkth.jld2")
    KLtokthplot = plot([SNCAKLkth, NCAKLkth],
        legend = :bottomright, color=[:orange :violetred4],
        linewidth = 2,
        label = ["SNCA" "NCA"],
        ylabel = "KL divergence",
        xlabel = L"k",
        size = (600, 300),
        dpi=600)
    KLtokthplot = scatter!([SNCAKLkth, NCAKLkth], color=[:orange :violetred4], labels = :none)
    return KLtokthplot
end

function Figure5()
    SNCA_Acc = load_object("res/TQ_SNCA_Acc.jld2")
    NCA_Acc = load_object("res/TQ_NCA_Acc.jld2")
    Euclidean_Acc = load_object("res/TQ_Euclidean_Acc.jld2")

    TQ_data = load_object("dat/TQ_data.jld2")
    x_matrix = Matrix{Float64}(transpose(TQ_data[:,1:(end-1)]));
    y = Vector{Int64}(TQ_data[:,end]);
    NNoptvalue = NNopt(x_matrix, y)

    NNAccplot = plot([[(sum(SNCA_Acc[i], dims=2)./10)[1] for i in 1:10],
        [(sum(NCA_Acc[i], dims=2)./10)[1] for i in 1:10],
        [(sum(Euclidean_Acc[i], dims=2)./10)[1] for i in 1:10]],
        ribbon = [(1.96/sqrt(10)).*[(std(SNCA_Acc[i], dims=2))[1] for i in 1:10],
        (1.96/sqrt(10)).*[(std(NCA_Acc[i], dims=2))[1] for i in 1:10],
        [(std(Euclidean_Acc[i], dims=2))[1] for i in 1:10]], fillalpha=0.3, linewidth = 2,
        legend = :bottomright, color=[:orange :violetred4 :black],
        #ylim = (0.5,0.65),
        label = ["SNCA" "NCA" "Euclidean"],
        ylabel = "1NN Accuracy",
        xlabel = L"n",
        xticks = 1:1:10,
        xformatter = i -> Int64(1000i),
        dpi=600,
        size=(600,300)
    )
    NNAccplot = plot!([1; 10], [NNoptvalue; NNoptvalue], lc=:black, linewidth = 2, linestyle=:dash, label="Optimal 1NN")
    NNAccplot = scatter!([[(sum(SNCA_Acc[i], dims=2)./10)[1] for i in 1:10],
        [(sum(NCA_Acc[i], dims=2)./10)[1] for i in 1:10]],
        color=[:orange :violetred4], labels = :none
    )

    SNCA_Acc_test = load_object("res/TQ_SNCA_Acc_test.jld2")
    NCA_Acc_test = load_object("res/TQ_NCA_Acc_test.jld2")
    Euclidean_Acc_test = load_object("res/TQ_Euclidean_Acc_test.jld2")
    NNopt_Acc_test = load_object("res/TQ_NNopt_Acc_test.jld2")

    NNtestAccplot = plot([[(sum(SNCA_Acc_test[i]))/60 for i in 1:10],
        [(sum(NCA_Acc_test[i]))/60 for i in 1:10],
        [(sum(Euclidean_Acc_test[1]))/60 for i in 1:10],
        [(sum(NNopt_Acc_test[2]))/60 for i in 1:10]],
        ribbon = [(1.96/sqrt(60)).*[std(SNCA_Acc_test[i]) for i in 1:10],
        (1.96/sqrt(60)).*[std(NCA_Acc_test[i]) for i in 1:10],
        (1.96/sqrt(60)).*[std(Euclidean_Acc_test[i]) for i in 1:10],
        (1.96/sqrt(60)).*[std(NNopt_Acc_test[i]) for i in 1:10]], fillalpha=0.3, linewidth = 2,
        legend = :bottomright, color=[:orange :violetred4 :black :black],
        linestyle = [:solid :solid :solid :dash],
        label = ["SNCA" "NCA" "Euclidean" "Optimal 1NN"],
        ylabel = "1NN Accuracy",
        xlabel = L"n",
        xticks = 1:1:10,
        xformatter = i -> Int64(1000i),
        dpi=600,
        size=(600,300)
        )
    NNtestAccplot = scatter!([[(sum(SNCA_Acc_test[i]))/60 for i in 1:10],
        [(sum(NCA_Acc_test[i]))/60 for i in 1:10]],
        color=[:orange :violetred4], labels = :none
    )

    plots_TQNN = plot(NNAccplot, NNtestAccplot, layout=(1,2), size=(1200,400), link=:all, margin=5mm)
    return plots_TQNN
end

function Figure6()
    SNCA_time = load_object("res/TQ_SNCA_time.jld2")
    NCA_time_pointwise = load_object("res/TQ_NCA_time_pointwise.jld2")

    timingplot = plot([[mean(SNCA_time[i])./60 for i in 1:10],
        [mean(NCA_time_pointwise[i])./60 for i in 1:10]],
        ribbon = [(1.96/sqrt(10)).*[std(SNCA_time[i])./60 for i in 1:10],
        (1.96/sqrt(10)).*[std(NCA_time_pointwise[i])./60 for i in 1:10]],
        fillalpha=0.3, linewidth = 2,
        legend = :topleft, color=[:orange :violetred4],
        label = ["SNCA" "NCA"],
        ylabel = "time (minutes)",
        xlabel = L"n",
        xticks = 1:1:10,
        xformatter = i -> Int64(1000i),
        dpi=600,
        size=(600,400)
    )
    timingplot = scatter!([[(mean(SNCA_time[i])./60) for i in 1:10],
        [(mean(NCA_time_pointwise[i])./60) for i in 1:10]],
        color=[:orange :violetred4], labels = :none
    )
    return timingplot
end

function Figure7()
    WF_data = load_object("dat/WF_data.jld2")
    A_SNCA = load_object("res/WF_SNCA_A.jld2")
    x_matrix = Matrix{Float64}(transpose(WF_data[:,1:(end-1)]));
    y = Vector{Int64}(WF_data[:,end]);

    Random.seed!(1234*5)
    x_partitions = balanced_kfold(y, 2)
    GR.setarrowsize(0.5)
    trans_points, densities, emp1 = plot2d(x_matrix[:,x_partitions[2]], y[x_partitions[2]], A_SNCA)
    Wfabpointsplot = scatter([bₗ[1] for bₗ in trans_points], [bₗ[2] for bₗ in trans_points],
        markersize = densities./25, 
        markerz = emp1,
        markerstrokewidth = 0.5,
        legend = false,
        xlim = (-20,70),
        ylim = (-7,13),
        colorbar = true,
        colorbar_title = L"\hat{q}(Y=1|X)",
        dpi=600)
    Wfabpointsplot = annotate!(13, -3.5, text("State 1", :black, :left, 8))
    Wfabpointsplot = plot!([12.3,8],[-3.3,-2.2],arrow = :closed, color=:black,linewidth=1,label="")
    Wfabpointsplot = annotate!(52, -1, text("State 2", :black, :left, 8))
    Wfabpointsplot = plot!([51.3,46.8],[-0.8,0.44],arrow = :closed, color=:black,linewidth=1,label="")

    SNCA_kNN = load_object("res/WF_SNCA_kNN.jld2")
    NCA_kNN = load_object("res/WF_NCA_kNN.jld2")
    Euclidean_kNN = load_object("res/WF_Euclidean_kNN.jld2")

    WfabkNNplot = plot([sum(SNCA_kNN,dims=2)./10,
        sum(NCA_kNN,dims=2)./10, 
        sum(Euclidean_kNN,dims=2)./10],
        ribbon = [(1.96/sqrt(10)).*[std(SNCA_kNN, dims=2)],
        (1.96/sqrt(10)).*[std(NCA_kNN, dims=2)],
        (1.96/sqrt(10)).*[std(Euclidean_kNN, dims=2)]], fillalpha=0.3, linewidth=2,
        legend=:bottomleft,
        label = ["SNCA" "NCA" "Euclidean"],
        color=[:orange :violetred4 :black],
        ylabel = "kNN Accuracy",
        xlabel = L"k",
        dpi=600)
    WfabkNNplot = scatter!([sum(SNCA_kNN,dims=2)./10,
        sum(NCA_kNN,dims=2)./10, 
        sum(Euclidean_kNN,dims=2)./10], 
        color=[:orange :violetred4 :black],
        label="")

    plots_Wfab = plot(Wfabpointsplot, WfabkNNplot, layout=(1,2), size=(1500,500), margin=5mm)
    return plots_Wfab
end