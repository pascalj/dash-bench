#!/usr/bin/env Rscript

source("plotlib.r")

measures <- measurements()
measures <- measures %>% mutate(total_threads = threads * processes)

bench_plot(measures, "total_threads", "time")

