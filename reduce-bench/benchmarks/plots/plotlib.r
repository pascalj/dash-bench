#!/usr/bin/env Rscript

library("tidyverse")

measurements <- function() {
  data <- readLines(file("stdin"))
  read_csv(data, col_names=c("n", "processes", "threads", "size", "time", "benchmark"))
}

bench_plot <- function(data, x, y, group = "benchmark") {

  out = glue::glue("{group}_{x}_{y}.png")

  benchmarks <- data %>%
    select_at(vars(x, y, group)) %>%
    group_by_at(vars(x, group)) %>%
    summarize_at(vars(y), mean)

  ggplot(benchmarks) +
    geom_path(aes_string(x=x, y=y, group=group, color=group)) +
    geom_point(aes_string(x=x, y=y, group=group, color=group, shape=group)) +
    theme_bw()

  ggsave(out)
  system(glue::glue("eog {out} &"))
}
