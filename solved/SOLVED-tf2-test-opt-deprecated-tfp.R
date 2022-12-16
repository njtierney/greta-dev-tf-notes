reprex::reprex({
  library(devtools)
  library(testthat)
  load_all(".")
  source(here::here("tests/testthat/helpers.R"))

  x <- rnorm(5, 2, 0.1)
  z <- variable(dim = 5)
  distribution(x) <- normal(z, 0.1)

  m <- model(z)

  opt(m, optimiser = adagrad_da())
  opt(m, optimiser = proximal_adagrad())
  opt(m, optimiser = proximal_gradient_descent())
},
wd = ".")
