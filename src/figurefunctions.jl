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
    vals = load_object("res/TQ_SNCA_vals.jld2")
    objvalue = load_object("res/TQ_SNCA_objvalue.jld2")
    A_SNCA = load_object("res/TQ_SNCA_A.jld2")
    # TQ_data = load_object("dat/TQ_data.jld2")
    # x_matrix = Matrix{Float64}(transpose(TQ_data[:,1:(end-1)]));
    # x = svectorscopy(x_matrix, Val(4));
    # y = Vector{Int64}(TQ_data[:,end]);
    # Euclid = SNCA(Matrix(I, 4, 4),x,y,objective=NCALog(),dims=Val(4));
    r = size(A_SNCA)[1]
    objvalueplot = plot(-vals[1:r],
        legend = :bottomright, color = :orange,
        linewidth = 2,
        label = :none,
        ylabel = "objective value",
        xlabel = L"r",
        tickfontsize = 7,
        ylim=(-0.6353, -0.6347),
        size = (600, 300),
        dpi=600)
    objvalueplot = plot!(-vals[1:r+1],
        color = :orange, linestyle = :dash,
        linewidth = 2, label = :none)
    objvalueplot = plot!([r, r], [-vals[r], -objvalue],
        color = :violetred4,
        linestyle = :dot,
        label = :none)
    objvalueplot = scatter!(-vals[1:r+1],
        color = :orange,
        label = "First stage")
    objvalueplot = scatter!([r], [-objvalue],
        color = :violetred4,
        label = "Second stage")
    # objvalueplot = plot!([-Inf,Inf], [-Euclid, -Euclid],
    #     color = :black, linestyle = :dash,
    #     linewidth = 2, label = :none)
    return objvalueplot
end
    

function Figure3()
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
        xlabel = L"A \mathbf{\textit{x}} [1]",
        ylabel = L"A \mathbf{\textit{x}} [2]",
        colorbar = false,
        tickfontsize = 7,
        dpi=600,
        size = (600,150))

    zerosklkth_SNCA = load_object("res/TQ_SNCA_klkth_0.jld2")
    SNCAKL = plot(zerosklkth_SNCA[1], linecolor = "black", alpha = 0.5, linewidth = 2,
        xlabel = L"k", ylabel = L"\textrm{KL \ divergence}", size = (600,150), dpi=600)
    SNCAKL = scatter!(zerosklkth_SNCA[1],
        markersize = zerosklkth_SNCA[3]./500,
        markerz = zerosklkth_SNCA[2],
        markerstrokewidth = 0.5,
        legend = false,
        ylim=(-0.05,0.8),
        colorbar = false,
        tickfontsize = 7,
        dpi=600,
        size = (600,150))
    h2 = scatter([0,0], [0,1], zcolor=[minimum(zerosklkth_SNCA[2]),maximum(zerosklkth_SNCA[2])],
        xlims=(1,1.1), clims=(0.1,0.8), framestyle=:none, label="", colorbar_title=L"\hat{q}(Y=1|X)", tickfontsize = 7, grid=false)
    l = @layout [grid(2, 1) a{0.035w}]
    plots_SNCA = plot(SNCAscatter, SNCAKL, h2, layout=l, dpi=1200)
    return plots_SNCA
end

function Figure4()
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
        ylim = (-70,20),
        xlabel = L"A \mathbf{\textit{x}} [1]",
        ylabel = L"A \mathbf{\textit{x}} [2]",
        colorbar = false,
        tickfontsize = 7,
        dpi=600,
        size = (600,150))

    zerosklkth_NCA = load_object("res/TQ_NCA_klkth_0.jld2")
    NCAKL = plot(zerosklkth_NCA[1], linecolor = "black", alpha = 0.5, linewidth = 2,
        xlabel = L"k", ylabel = L"\textrm{KL \ divergence}", size = (600,150), dpi=600)
    NCAKL = scatter!(zerosklkth_NCA[1],
        markersize = zerosklkth_NCA[3]./500,
        markerz = zerosklkth_NCA[2],
        markerstrokewidth = 0.5,
        legend = false,
        ylim=(-0.05,0.8),
        colorbar = false,
        tickfontsize = 7,
        dpi=600,
        size = (600,150))
    h2 = scatter([0,0], [0,1], zcolor=[minimum(zerosklkth_NCA[2]),maximum(zerosklkth_NCA[2])],
        xlims=(1,1.1), clims=(0.1,0.8), framestyle=:none, label="", colorbar_title=L"\hat{q}(Y=1|X)", tickfontsize = 7, grid=false)
    l = @layout [grid(2, 1) a{0.035w}]
    plots_NCA = plot(NCAscatter, NCAKL, h2, layout=l, dpi=1200)
    return plots_NCA
end

function Figure5()
    SNCAKLkth = load_object("res/TQ_SNCA_klkth.jld2")
    NCAKLkth = load_object("res/TQ_NCA_klkth.jld2")
    KLtokthplot = plot([SNCAKLkth, NCAKLkth],
        legend = :bottomright, color=[:orange :violetred4],
        linewidth = 2,
        label = [L"\textrm{SNCA}" L"\textrm{NCA}"],
        ylabel = L"\textrm{KL \ divergence}",
        xlabel = L"k",
        tickfontsize = 7,
        ylim=(-0.02,0.5),
        size = (600, 300),
        dpi=600)
    KLtokthplot = scatter!([SNCAKLkth, NCAKLkth], color=[:orange :violetred4], labels = :none)
    return KLtokthplot
end

function Figure6()
    SNCA_Acc = load_object("res/TQ_SNCA_Acc.jld2")
    NCA_Acc = load_object("res/TQ_NCA_Acc.jld2")
    Euclidean_Acc = load_object("res/TQ_Euclidean_Acc.jld2")

    TQ_data = load_object("dat/TQ_data.jld2")
    x_matrix = Matrix{Float64}(transpose(TQ_data[:,1:(end-1)]));
    y = Vector{Int64}(TQ_data[:,end]);
    NNoptvalue = NNopt(x_matrix, y)

    NNAccplot = plot([1; 10], [NNoptvalue; NNoptvalue], 
        lc=:black, linewidth = 2, linestyle=:dash, 
        label=L"\textrm{Optimal \ 1NN}",
        legend = :bottomright, 
        ylabel = L"\textrm{1NN \ Accuracy}",
        xlabel = L"n",
        xticks = 1:1:10,
        xformatter = i -> Int64(1000i),
        ylim=(0.547,0.65),
        yticks=(0.55:0.02:0.65),
        dpi=600,
        size=(600,300)
    )
    NNAccplot = plot!([(sum(SNCA_Acc[i], dims=2)./10)[1] for i in 1:10],
        ribbon = (1.96/sqrt(10)).*[(std(SNCA_Acc[i], dims=2))[1] for i in 1:10],
        fillalpha=0.3, linewidth = 2,
        color = :orange,
        label = L"\textrm{SNCA}"
    )
    NNAccplot = plot!([(sum(NCA_Acc[i], dims=2)./10)[1] for i in 1:10],
        ribbon = (1.96/sqrt(10)).*[(std(NCA_Acc[i], dims=2))[1] for i in 1:10],
        fillalpha=0.3, linewidth = 2,
        color = :violetred4,
        label = L"\textrm{NCA}"
    )
    NNAccplot = plot!([(sum(Euclidean_Acc[i], dims=2)./10)[1] for i in 1:10],
        ribbon = [(std(Euclidean_Acc[i], dims=2))[1] for i in 1:10],
        fillalpha=0.3, linewidth = 2,
        color = :black,
        label = L"\textrm{Euclidean}"
    )
    NNAccplot = scatter!([[(sum(SNCA_Acc[i], dims=2)./10)[1] for i in 1:10],
        [(sum(NCA_Acc[i], dims=2)./10)[1] for i in 1:10]],
        color=[:orange :violetred4], labels = :none
    )

    SNCA_Acc_test = load_object("res/TQ_SNCA_Acc_test.jld2")
    NCA_Acc_test = load_object("res/TQ_NCA_Acc_test.jld2")
    Euclidean_Acc_test = load_object("res/TQ_Euclidean_Acc_test.jld2")
    NNopt_Acc_test = load_object("res/TQ_NNopt_Acc_test.jld2")

    NNtestAccplot = plot([(sum(NNopt_Acc_test[i]))/60 for i in 1:10],
        #[(sum(NCA_Acc_test[i]))/60 for i in 1:10],
        #[(sum(Euclidean_Acc_test[1]))/60 for i in 1:10],
        #[(sum(NNopt_Acc_test[2]))/60 for i in 1:10]],
        ribbon = (1.96/sqrt(60)).*[std(NNopt_Acc_test[i]) for i in 1:10],
        #(1.96/sqrt(60)).*[std(NCA_Acc_test[i]) for i in 1:10],
        #(1.96/sqrt(60)).*[std(Euclidean_Acc_test[i]) for i in 1:10],
        #(1.96/sqrt(60)).*[std(NNopt_Acc_test[i]) for i in 1:10]],
        fillalpha=0.3, linewidth = 2,
        legend = :bottomright, color=:black,
        linestyle = :dash,
        label = L"\textrm{Optimal \ 1NN}", #L"\textrm{NCA}" L"\textrm{Euclidean}" L"\textrm{Optimal \ 1NN}"],
        ylabel = L"\textrm{1NN \ Accuracy}",
        xlabel = L"n",
        xticks = 1:1:10,
        xformatter = i -> Int64(1000i),
        ylim=(0.547,0.65),
        yticks=(0.55:0.02:0.65),
        dpi=600,
        size=(600,300)
    )
    NNtestAccplot = plot!([(sum(SNCA_Acc_test[i]))/60 for i in 1:10],
        ribbon = (1.96/sqrt(60)).*[std(SNCA_Acc_test[i]) for i in 1:10],
        fillalpha=0.3, linewidth = 2,
        legend = :bottomright, color=:orange,
        linestyle = :solid,
        label = L"\textrm{SNCA}"
    ) 
    NNtestAccplot = plot!([(sum(NCA_Acc_test[i]))/60 for i in 1:10],
        ribbon = (1.96/sqrt(60)).*[std(NCA_Acc_test[i]) for i in 1:10],
        fillalpha=0.3, linewidth = 2,
        legend = :bottomright, color=:violetred4,
        linestyle = :solid,
        label = L"\textrm{NCA}"
    )
    NNtestAccplot = plot!([(sum(Euclidean_Acc_test[1]))/60 for i in 1:10],
        ribbon = (1.96/sqrt(60)).*[std(Euclidean_Acc_test[i]) for i in 1:10],
        fillalpha=0.3, linewidth = 2,
        legend = :bottomright, color=:black,
        linestyle = :solid,
        label = L"\textrm{Euclidean}"
    )
    NNtestAccplot = scatter!([[(sum(SNCA_Acc_test[i]))/60 for i in 1:10],
        [(sum(NCA_Acc_test[i]))/60 for i in 1:10]],
        color=[:orange :violetred4], labels = :none
    )

    plots_TQNN = plot(NNAccplot, NNtestAccplot, layout=(1,2), size=(1200,400), margin=5mm)
    return plots_TQNN
end

function Figure7()
    SNCA_time = load_object("res/TQ_SNCA_time.jld2")
    NCA_time_pointwise = load_object("res/TQ_NCA_time_pointwise.jld2")

    timingplot = plot([mean(SNCA_time[i])./60 for i in 1:10],
        ribbon = (1.96/sqrt(10)).*[std(SNCA_time[i])./60 for i in 1:10],
        fillalpha=0.3, linewidth = 2,
        legend = :topleft, color=:orange,
        label = L"\textrm{SNCA}",
        ylabel = L"\textrm{time \ (minutes)}",
        xlabel = L"n",
        xticks = 1:1:10,
        xformatter = i -> Int64(1000i),
        ylim=(-10,300),
        dpi=600,
        size=(600,400)
    )
    timingplot = plot!([mean(NCA_time_pointwise[i])./60 for i in 1:10],
        ribbon = (1.96/sqrt(10)).*[std(NCA_time_pointwise[i])./60 for i in 1:10],
        fillalpha=0.3, linewidth = 2,
        color=:violetred4,
        label = L"\textrm{NCA}"
    )
    timingplot = scatter!([[(mean(SNCA_time[i])./60) for i in 1:10],
        [(mean(NCA_time_pointwise[i])./60) for i in 1:10]],
        color=[:orange :violetred4], labels = :none
    )
    return timingplot
end

function Figure8()
    WF_data = load_object("dat/WF_data.jld2")
    A_SNCA = load_object("res/WF_SNCA_A.jld2")
    x_matrix = Matrix{Float64}(transpose(WF_data[:,1:(end-1)]));
    y = Vector{Int64}(WF_data[:,end]);

    vals = load_object("res/WF_SNCA_vals.jld2")
    objvalue = load_object("res/WF_SNCA_objvalue.jld2")
    r = size(A_SNCA)[1]
    objvalueplot = plot(-vals[1:r],
        legend = :bottomright, color = :orange,
        linewidth = 2,
        label = :none,
        ylabel = "objective value",
        xlabel = L"r",
        tickfontsize = 7,
        ylim=(-0.611, -0.605),
        size = (600, 300),
        dpi=600)
    objvalueplot = plot!(-vals[1:r+1],
        color = :orange, linestyle = :dash,
        linewidth = 2, label = :none)
    objvalueplot = plot!([r, r], [-vals[r], -objvalue],
        color = :violetred4,
        linestyle = :dot,
        label = :none)
    objvalueplot = scatter!(-vals[1:r+1],
        color = :orange,
        label = "First stage")
    objvalueplot = scatter!([r], [-objvalue],
        color = :violetred4,
        label = "Second stage")

    Random.seed!(1234*1)
    x_partitions = balanced_kfold(y, 2)
    GR.setarrowsize(0.5)
    trans_points, densities, emp1 = plot2d(x_matrix[:,x_partitions[1]], y[x_partitions[1]], A_SNCA)
    Wfabpointsplot = scatter([-bₗ[1] for bₗ in trans_points], [-bₗ[2] for bₗ in trans_points],
        markersize = densities./25, 
        markerz = emp1,
        markerstrokewidth = 0.5,
        legend = false,
        xlim = (-40,130),
        ylim = (-7,25),
        colorbar = true,
        colorbar_title = L"\hat{q}(Y=1|X)",
        xlabel = L"A \mathbf{\textit{x}} [1]",
        ylabel = L"A \mathbf{\textit{x}} [2]",
        tickfontsize = 10,
        dpi=600)
    Wfabpointsplot = annotate!(31, -3.9, text(L"\textrm{State \ 1}", :black, :left, 10))
    Wfabpointsplot = plot!([30,20],[-4,-3],arrow = :closed, color=:black,linewidth=1,label="")
    Wfabpointsplot = annotate!(95, 7, text(L"\textrm{State \ 2}", :black, :left, 10))
    Wfabpointsplot = plot!([94,85],[7,8.7],arrow = :closed, color=:black,linewidth=1,label="")

    SNCA_kNN = load_object("res/WF_SNCA_kNN.jld2")
    NCA_kNN = load_object("res/WF_NCA_kNN.jld2")
    Euclidean_kNN = load_object("res/WF_Euclidean_kNN.jld2")

    WfabkNNplot = plot(sum(SNCA_kNN,dims=2)./10,
        ribbon = (1.96/sqrt(10)).*std(SNCA_kNN, dims=2),
        label = L"\textrm{SNCA}",
        color = :orange,
        fillalpha = 0.3,
        linewidth = 2,
        legend=:bottomleft,
        ylabel = L"k\textrm{NN \ Accuracy}",
        xlabel = L"k",
        ylim=(0.525, 0.7),
        tickfontsize = 10,
        xguidefontsize = 12,
        yguidefontsize = 12,
        legendfontsize = 10,
        dpi=600)
    WfabkNNplot = plot!(sum(NCA_kNN,dims=2)./10,
        ribbon = (1.96/sqrt(10)).*std(NCA_kNN, dims=2),
        label = L"\textrm{NCA}",
        color = :violetred4,
        fillalpha = 0.3,
        linewidth = 2)    
    WfabkNNplot = plot!(sum(Euclidean_kNN,dims=2)./10,
        ribbon = (1.96/sqrt(10)).*std(Euclidean_kNN, dims=2),
        label = L"\textrm{Euclidean}",
        color = :black,
        fillalpha = 0.3,
        linewidth = 2)
    WfabkNNplot = scatter!([sum(SNCA_kNN,dims=2)./10,
        sum(NCA_kNN,dims=2)./10, 
        sum(Euclidean_kNN,dims=2)./10], 
        color=[:orange :violetred4 :black],
        label="")

    #plots_Wfab = plot(Wfabpointsplot, WfabkNNplot, layout=(1,2), size=(1500,500), margin=6mm)
    l = @layout [
        a{0.5w} [grid(2,1, heights = [0.4, 0.6])]
    ]
    plots_Wfab = plot(Wfabpointsplot, objvalueplot, WfabkNNplot, layout=l, size=(1500,500), margin=6mm)

    return plots_Wfab
end