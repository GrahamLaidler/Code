# Stochastic Neighbourhood Components Analysis

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6923145.svg)](https://doi.org/10.5281/zenodo.6923145)

This repository contains data and code used in the analysis of SNCA.

To use, clone the repository. Refer to the [instructions](https://pkgdocs.julialang.org/v1.2/environments/#Using-someone-else's-project-1) for setting up Julia environments.

The following files should be used to run the analysis:
- TQanalysis.jl: Perform experiments using data from a tandem queueing simualtion.
- WFanalysis.jl: Perform experiments using data from a wafer fab simualtion.
- figures.jl: Generate figures from results of the previous two files.
