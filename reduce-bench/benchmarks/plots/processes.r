#!/usr/bin/env Rscript

source("plotlib.r")

measures = measurements()
bench_plot(measures, "processes", "time")
