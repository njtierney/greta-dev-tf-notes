# errors in test_inference.R

## main issue is that this MCMC should be erroring...but it isn't!

## Error 1 - SOLVED with 1e12...to make the distributions farther apart
devtools::load_all(".")
source("tests/testthat/helpers.R")
# set up for numerical rejection of initial location
x <- rnorm(10000, 1e12, 1)
z <- normal(-1e12, 1e-12)
distribution(x) <- normal(z, 1e12)
m <- model(z, precision = "single")

out <- get_output(
  draws <- mcmc(m, n_samples = 10, warmup = 0, pb_update = 10)
)

cat(out)

expect_match(out, "100% bad")

## error 2 - we expect this to error but it doesn't
## OK so it appears that this might be due to the fact that are we using
## single precision, but it looks like the dag defines it as double precision
## we need to explore where each dag is defined so we can pass the appropriate
## prevision - take a look at where previously on_graph was called in
## the TF1 releaase of greta. Since we don't use on_graph anymore, we need
## some way of handling that carefully and appropriately
## so take a look at the use of on_graph and the on.exit pattern used to
## manage precision
  draws <- mcmc(m,
       chains = 1,
       n_samples = 1,
       warmup = 0,
       verbose = FALSE,
       initial_values = initials(z = 1e20)
  )


## error 3 - we expect this to have really bad proposals, but no!
# really bad proposals
x <- rnorm(100000, 1e12, 1)
z <- normal(-1e12, 1e-12)
distribution(x) <- normal(z, 1e-12)
m <- model(z, precision = "single")
mcmc(m, chains = 1, n_samples = 1, warmup = 0, verbose = FALSE)

## error 4 - proposals that are fine, but rejected anyway
z <- normal(0, 1)
m <- model(z, precision = "single")
expect_ok(mcmc(m,
               hmc(
                 epsilon = 100,
                 Lmin = 1,
                 Lmax = 1
               ),
               chains = 1,
               n_samples = 5,
               warmup = 0,
               verbose = FALSE
))

## error 4 - mcmc supports rwmh sampler with normal proposals
## Error in py_call_impl(callable, dots$args, dots$keywords) :
## RuntimeError: Evaluation error: ValueError: Cannot reshape a tensor with 0
## elements to shape [1] (1 elements) for '{{node Reshape}} =
## Reshape[T=DT_DOUBLE, Tshape=DT_INT32](Mul, Reshape/shape)'
## with input shapes: [0], [1] and with input tensors computed as partial
## shapes: input[1] = [1].
x <- normal(0, 1)
m <- model(x)
expect_ok(draws <- mcmc(m,
                        sampler = rwmh("normal"),
                        n_samples = 100, warmup = 100,
                        verbose = FALSE
))

## error 5 - mcmc supports rwmh sampler with uniform proposals
set.seed(5)
x <- uniform(0, 1)
m <- model(x)
expect_ok(draws <- mcmc(m,
                        sampler = rwmh("uniform"),
                        n_samples = 100, warmup = 100,
                        verbose = FALSE
))

## error 6 - mcmc supports slice sampler with single precision models
set.seed(5)
x <- uniform(0, 1)
m <- model(x, precision = "single")
expect_ok(draws <- mcmc(m,
                        sampler = slice(),
                        n_samples = 100, warmup = 100,
                        verbose = FALSE
))

## error 7 - mcmc doesn't support slice sampler with double precision models
set.seed(5)
x <- uniform(0, 1)
m <- model(x, precision = "double")
## This should error - but it gives a definitely different error!
# expect_snapshot_error(
draws <- mcmc(m,
              sampler = slice(),
              n_samples = 100, warmup = 100,
              verbose = FALSE
)
# )

## error 8 - numerical issues are handled in mcmc - does not error
# this should have a cholesky decomposition problem at some point
alpha <- normal(0, 1)
x <- matrix(rnorm(6), 3, 2)
y <- t(rnorm(3))
z <- alpha * x
sigma <- z %*% t(z)
distribution(y) <- multivariate_normal(zeros(1, 3), sigma)
m <- model(alpha)

# running with bursts should error informatively
# expect_snapshot_error(
draws <- mcmc(m, verbose = FALSE)
# )

## error 9 - mcmc works in parallel
## issues with mcmc failing with future - first try without future
# turn off- need to restart R
devtools::load_all(".")
source("tests/testthat/helpers.R")
m <- model(normal(0, 1))

# one chain
expect_ok(draws <- mcmc(m,
                        warmup = 10, n_samples = 10,
                        chains = 1,
                        verbose = FALSE
))

# now try with future - it fails :(

op <- future::plan()
# put the future plan back as we found it
withr::defer(future::plan(op))
future::plan(future::sequential)

# works for sequential
# one chain
expect_ok(draws <- mcmc(m,
                        warmup = 10, n_samples = 10,
                        chains = 1,
                        verbose = FALSE
))

future::plan(future::multisession)
# one chain
expect_ok(draws <- mcmc(m,
                        warmup = 10, n_samples = 10,
                        chains = 1,
                        verbose = FALSE
))

## Error 10 - parallel reporting works
## this errors with:
## Error in py_call_impl(callable, dots$args, dots$keywords) :
## ValueError: Attempt to convert a value (None) with an unsupported type
## (<class 'NoneType'>) to a Tensor.

m <- model(normal(0, 1))

op <- future::plan()
# put the future plan back as we found it
withr::defer(future::plan(op))
future::plan(future::multisession)

# should report each sampler's progress with a fraction
out <- get_output(. <- mcmc(m, warmup = 50, n_samples = 50, chains = 2))
expect_match(out, "2 samplers in parallel")
expect_match(out, "50/50")
