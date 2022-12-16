# test_variable.R ------
reprex::reprex({
  devtools::load_all(".")
  source("tests/testthat/helpers.R")
  ## variable() with universal bounds can be sampled correctly
  x <- rnorm(3, 0, 10)
  mu <- variable(
    lower = 2,
    upper = 6
  )
  distribution(x) <- normal(mu, 1)
  m <- model(mu)
  draws <- mcmc(m, n_samples = 100, warmup = 1, verbose = FALSE)

  samples <- as.matrix(draws)
  above_lower <- sweep(samples, 2, 2, `>=`)
  below_upper <- sweep(samples, 2, 6, `<=`)

  expect_true(all(above_lower & below_upper))

  ## variable() with vectorised bounds can be sampled correctly
  x <- rnorm(3, 0, 10)
  lower <- c(-3, -1, 2)
  upper <- c(0, 2, 3)
  mu <- variable(
    lower = lower,
    upper = upper
  )
  distribution(x) <- normal(mu, 1)
  m <- model(mu)
  draws <- mcmc(m, n_samples = 100, warmup = 1, verbose = FALSE)

  samples <- as.matrix(draws)
  above_lower <- sweep(samples, 2, lower, `>=`)
  below_upper <- sweep(samples, 2, upper, `<=`)

  expect_true(all(above_lower & below_upper))

},
wd = ".")
