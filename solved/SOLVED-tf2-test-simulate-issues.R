# test_simulate.R ----
reprex::reprex({
  devtools::load_all(".")
  source("tests/testthat/helpers.R")
  # fix variable
  a <- normal(0, 1)
  y <- normal(a, 1)
  m <- model(y)

  # the samples should be the same if the seed is the same
  one <- simulate(m, seed = 12345)
  two <- simulate(m, seed = 12345)
  expect_identical(one, two)
},
wd = ".")
