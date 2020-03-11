#!/usr/bin/env Rscript

library("tidyverse")

x = "processes"
y = "time"
group = "benchmark"
out = glue::glue("{group}_{x}_{y}.png")

measurements <- readLines(file("stdin"))

benchmarks <- read_csv(measurements, col_names=c("n", "processes", "threads", "size", "time", "benchmark"))



benchmarks <- benchmarks %>% group_by(processes, threads, benchmark) %>% summarize(time=mean(time))

print(benchmarks)
ggplot(benchmarks) +
  geom_path(aes_string(x=x, y=y, group=group))
  # geom_point(aes_string(x=x, y=y, group=group))

ggsave(out)
