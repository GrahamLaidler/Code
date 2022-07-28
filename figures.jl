include("src/functionswrapper.jl")
include("src/figurefunctions.jl")
using Plots, Plots.PlotMeasures, LaTeXStrings

## Figure 2
Figure2()  #plot(Figure2())
savefig("fig/Fig2.png")
println("Finished figure 2")

## Figure 3
Figure3()  #plot(Figure2())
savefig("fig/Fig3.png")
println("Finished figure 3")

## Figure 4
Figure4()  #plot(Figure2())
savefig("fig/Fig4.png")
println("Finished figure 4")

## Figure 5
Figure5()  #plot(Figure2())
savefig("fig/Fig5.png")
println("Finished figure 5")

## Figure 6
Figure6()  #plot(Figure2())
savefig("fig/Fig6.png")
println("Finished figure 6")