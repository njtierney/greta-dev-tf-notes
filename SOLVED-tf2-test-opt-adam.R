reprex::reprex({
library(devtools)
library(testthat)
load_all(".")
source(here::here("tests/testthat/helpers.R"))

x <- rnorm(5, 2, 0.1)
z <- variable(dim = 5)
distribution(x) <- normal(z, 0.1)

m <- model(z)

optimisers <- lst(
  gradient_descent,
  adadelta,
  adagrad,
  adam,
  ftrl,
  rms_prop
)

opt_df <- opt_df_run(optimisers, m, x)
tidied_opt <- tidy_optimisers(opt_df, tolerance = 1e-2)

expect_true(all(tidied_opt$convergence == 0))
expect_true(all(tidied_opt$iterations <= 200))
expect_true(all(tidied_opt$close_to_truth))

tidied_opt %>%
  filter(!close_to_truth) %>%
  select(opt,
         par_x_diff,
         par,
         convergence,
         iterations,
         x_val) %>%
  unnest_longer(col = c(par_x_diff, par, x_val)) %>%
  knitr::kable()
},
wd = ".")
