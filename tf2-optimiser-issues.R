## test_opt.R
## The optimisers fail because they need to be changed over, which is happening in:
# https://github.com/greta-dev/greta/pull/550/
# But I to properly that branch into the tf2 branch
# here's a demonstration of one of the issues
reprex::reprex({
  devtools::load_all(".")
  source("tests/testthat/helpers.R")
  sd <- runif(5)
  x <- rnorm(5, 2, 0.1)
  z <- variable(dim = 5)
  distribution(x) <- normal(z, sd)

  m <- model(z)
  o <- opt(m, hessian = TRUE)
},
wd = ".")
