## General notes ====
# there are a couple of distributions that fail a few different tests across
# greta. These seem to be the multivariate ones: multinomial, dirichlet, categorical
# and also later in other places, wishart. I think theres some issue with
# the distributions in general
## ====
# usethis::pr_fetch(534)
devtools::load_all(".")
source("tests/testthat/helpers.R")

## There's an interesting clue here, in test_joint.R, I was noticing a pattern
# and here's a small example of some distribution code that fails.

sample_distribution(uniform(0,1))

# which seems crazy, given that uniform should be easy to sample?
# can we run mcmc on it?

uniform_samples <- mcmc(model(uniform(0,1)))

# what the heck

# but we can sample a normal.

normal_samples <- mcmc(model(normal(0,1)))

# Good.

# so this is happening at a pretty deep place I guess
# debugonce(mcmc)
options(error = recover)

# Enter a frame number, or 0 to exit
#
# 1: mcmc(model(uniform(0, 1)))
# 2: inference.R#201: with(tf$device(compute_options), {
# trace_batch_size
# 3: with.python.builtin.object(tf$device(compute_options), {
#   trace_batch
#   4: tryCatch(force(expr), finally = {
#     data$`__exit__`(NULL, NULL, NULL)
#   }
#   5: tryCatchList(expr, classes, parentenv, handlers)
#   6: force(expr)
#   7: inference.R#258: lapply(initial_values_split, build_sampler, sampler, m
#   8: FUN(X[[i]], ...)
#   9: utils.R#554: sampler$class$new(initial_values, model, sampler$parameter
#   10: initialize(...)
#   11: inference_class.R#302: super$initialize(initial_values = initial_values
#   12: inference_class.R#51: self$set_initial_values(initial_values)
#   13: inference_class.R#154: lapply(init_list, self$check_initial_values)
#   14: FUN(X[[i]], ...)

# So this is happening at
# self$check_initial_values

# self$check_initial_values does the following
  # checks the values are intialised/set - for those that aren't
  # it samples a small value from rnorm
  # then it checks that the values are valid with `self$valid_parameters`

  # `self$valid_parameters` does the following
      # valid_parameters = function(parameters) {
      # dag <- self$model$dag
      # tf_parameters <- fl(array(
      #   data = parameters,
      #   dim = c(1, length(parameters))
      # ))
      # ld <- lapply(
      #   dag$tf_log_prob_function(tf_parameters),
      #   as.numeric
      # )
      # is.finite(ld$adjusted) && is.finite(ld$unadjusted)
  # my understanding of this (`tf_log_prob_function`) is
      # make a TF array of the parameters - which are the rnorm values
      # apply the log prob function on it
        # this calls self$generate_log_prob_function()
        # which does a lot of code to fit in here.
      # check that the log prob values aren't infinite

# the issue is that the values returned by `tf_log_prob_function`
# are -Inf or Inf for when sampling from uniform, but not for normal
# so when we run
## uniform_samples <- mcmc(model(uniform(0,1)))
## and trace this through to the `valid_parameters` function
## here `tf_parameters` contains:
## tf.Tensor([[-0.06387527]], shape=(1, 1), dtype=float64)
# and then  `dag$tf_log_prob_function(tf_parameters)`
# returns
# $adjusted
# tf.Tensor([-inf], shape=(1), dtype=float64)
#
# $unadjusted
# tf.Tensor([-inf], shape=(1), dtype=float64)

# But this is what happens with `normal`
# normal_samples <- mcmc(model(normal(0,1)))
# `tf_parameters`
# tf.Tensor([[-0.11163191]], shape=(1, 1), dtype=float64)
# lapply(
#   dag$tf_log_prob_function(tf_parameters),
#   as.numeric
# )

# gives

# $adjusted
# [1] -0.9251694
#
# $unadjusted
# [1] -0.9251694

## So this tells us that something is going on with `tf_log_prob_function`
## Need to investigate more there

####

# popping a browser in `generate_log_prob_function()`, loading all again...

uniform_samples <- mcmc(model(uniform(0,1)))

###
# Tracing the function through the debugger I believe `tf_parameters` is the free state.
#
# > -0.06387527 would be an invalid suggestion for the actual parameter of U(0, 1), so it should return Inf when checking the density.
#
# Yup that makes sense
#
# > But that should never happen; the `rnorm()` bit should be generating proposals on the _free state_, and then inside `tf_log_prob_function()` those should be transformed into appropriate parameter values (so converted to the unit interval via a logit transform in the case of uniform). It's worth double checking that that is happening. there was some funny stuff with the chained bijectors (which do this transformation) before, so maybe that's where the bug is for this one?
#
#   Hmmm! So I wonder if that is supposed to happen inside of `evaluate_density`?
#
#   Basically the parts inside of `tf_log_prob_function` are:
#
#   ```r
# # temporarily define a new environment
# tfe_old <- self$tf_environment
# on.exit(self$tf_environment <- tfe_old)
# tfe <- self$tf_environment <- new.env()
# # put the free state in the environment, and build out the tf graph
# tfe$free_state <- free_state
#
# # we now make all of the operations define themselves now
# self$define_tf()
# # define the densities
# self$define_joint_density()
# ```
#
# And then `define_joint_density()` is
#
# ```r
# define_joint_density = function() {
#   # browser()
#   tfe <- self$tf_environment
#
#   # get all distribution nodes that have a target
#   distribution_nodes <- self$node_list[self$node_types == "distribution"]
#   target_nodes <- lapply(distribution_nodes, member, "get_tf_target_node()")
#   has_target <- !vapply(target_nodes, is.null, FUN.VALUE = TRUE)
#   distribution_nodes <- distribution_nodes[has_target]
#   target_nodes <- target_nodes[has_target]
#
#   # get the densities, evaluated at these targets
#   densities <- mapply(self$evaluate_density,
#                       distribution_nodes,
#                       target_nodes,
#                       SIMPLIFY = FALSE
#   )
#   ```
#
#   Then stepping into `evaluate_density` - it looks like:
#
#     ```r
#   evaluate_density = function(distribution_node, target_node) {
#     tfe <- self$tf_environment
#
#     parameter_nodes <- distribution_node$parameters
#
#     # get the tensorflow objects for these
#     distrib_constructor <- self$get_tf_object(distribution_node)
#     tf_target <- self$get_tf_object(target_node)
#     tf_parameter_list <- lapply(parameter_nodes, self$get_tf_object)
#
#     # execute the distribution constructor functions to return a tfp
#     # distribution object
#     tfp_distribution <- distrib_constructor(tf_parameter_list, dag = self)
#     # browser()
#     self$tf_evaluate_density(tfp_distribution,
#                              tf_target,
#                              truncation = distribution_node$truncation,
#                              bounds = distribution_node$bounds
#     )
#   },
#   ```
#
#   debugging this, and running
#
#   ```r
#   self$tf_evaluate_density(tfp_distribution,
#                            tf_target,
#                            truncation = distribution_node$truncation,
#                            bounds = distribution_node$bounds
#   )
#   ```
#
#   Gives
#
#   ```r
#   tf.Tensor([[[-inf]]], shape=(1, 1, 1), dtype=float64)
#   ```
#
#   So, then, stepping into `self$tf_evaluate_density`
#
#   We get:
#
#     ```r
#   tf_evaluate_density = function(tfp_distribution,
#                                  tf_target,
#                                  truncation = NULL,
#                                  bounds = NULL) {
#
#     # get the uncorrected log density
#     ld <- tfp_distribution$log_prob(tf_target)
#
#     # if required, calculate the log-adjustment to the truncation term of
#     # the density function i.e. the density of a distribution, truncated
#     # between a and b, is the non truncated density, divided by the integral
#     # of the density function between the truncation bounds. This can be
#     # calculated from the distribution's CDF
#     if (!is.null(truncation)) {
#       lower <- truncation[[1]]
#       upper <- truncation[[2]]
#
#       if (all(lower == bounds[1])) {
#
#         # if only upper is constrained, just need the cdf at the upper
#         offset <- tfp_distribution$log_cdf(fl(upper))
#       } else if (all(upper == bounds[2])) {
#
#         # if only lower is constrained, get the log of the integral above it
#         offset <- tf$math$log(fl(1) - tfp_distribution$cdf(fl(lower)))
#       } else {
#
#         # if both are constrained, get the log of the integral between them
#         offset <- tf$math$log(tfp_distribution$cdf(fl(upper)) -
#                                 tfp_distribution$cdf(fl(lower)))
#       }
#
#       ld <- ld - offset
#     }
#
#
#     ld
#   },
#
#   ```
#
#   So debugging this, `tf_target` is
#
#   ```r
#   Browse[5]> tf_target
#   tf.Tensor([[[1.47212097]]], shape=(1, 1, 1), dtype=float64)
#   ```
#
#   And then we're at the point where we calculate the `log_prob`
#
# ```r
# ld <- tfp_distribution$log_prob(tf_target)
# ```
#
# And it seems that perhaps `tf_target` hasn't been transformed yet? Because then we get
#
#   ```r
#   Browse[5]> tfp_distribution$log_prob(tf_target)
#   ```
#
#   ```r
#   tf.Tensor([[[-inf]]], shape=(1, 1, 1), dtype=float64)
#   ```
#
#   And having a bit of a poke around https://github.com/greta-dev/greta/blob/master/R/dag_class.R
#
#   It looks like this hasn't changed much from the TF2 branch.
#
# So I'm not quite sure how to solve this issue of ensuring that the values are appropriately transformed? I'm most likely missing something!
### UP TO HERE
###


#' free state is
#> tf.Tensor([[-0.06387527]], shape=(1, 1), dtype=float64)

# then we do

## temporarily define a new environment
# tfe_old <- self$tf_environment
# on.exit(self$tf_environment <- tfe_old)
# tfe <- self$tf_environment <- new.env()
## put the free state in the environment, and build out the tf graph
#tfe$free_state <- free_state
#
## we now make all of the operations define themselves now
# self$define_tf()
## define the densities
# self$define_joint_density()

# free_state is `tf.Tensor([[-0.06387527]], shape=(1, 1), dtype=float64)`

# it seems that the issue might be in `define_joint_density()`

# ok and then running

# mapply(self$evaluate_density,
#        distribution_nodes,
#        target_nodes,
#        SIMPLIFY = FALSE)
#
# gives
#
# $node_54966a45
# tf.Tensor([[[-inf]]], shape=(1, 1, 1), dtype=float64)

# so why the hell does that happen?

# let's look inside `self$evaluate_density()`
# which is placing a browser inside `evaluate_density`

# OK, and then running
#
# self$tf_evaluate_density(tfp_distribution,
#                          tf_target,
#                          truncation = distribution_node$truncation,
#                          bounds = distribution_node$bounds
# )

# returns the Inf thing
# so let's jump into `tf_evaluate_density()`

# bingo
# tf_target
# is
# tf.Tensor([[[1.48403661]]], shape=(1, 1, 1), dtype=float64)
# and
# tfp_distribution$log_prob(tf_target)
# is
# tf.Tensor([[[-inf]]], shape=(1, 1, 1), dtype=float64)

# So we're feeding tfp_distribution the wrong thing, or something?
# the log_prob shouldn't be that
# or is the value being provided in there out of the scope of what it should be?
# like it's 1.4, and the uniform is 0-1, so it has no density there?
# But then shouldn't it be positive, not negative?

# what is it in the `normal` case?

#
normal_samples <- mcmc(model(normal(0,1)))

# tf_target
# is
# tf.Tensor([[[0.02287592]]], shape=(1, 1, 1), dtype=float64)
# and
# tfp_distribution$log_prob(tf_target)
# is
# tf.Tensor([[[-0.91920019]]], shape=(1, 1, 1), dtype=float64)

# As far as I can tell, that is

# (4 errors) test_distributions
# Could not find reasonable starting values after 20 attempts.
# (compare_distributions fails)

devtools::load_all(".")
source("tests/testthat/helpers.R")

## error 1
# multivariate discrete
y <- extraDistr::rmnom(5, size = 4, prob = runif(3))
p <- uniform(0, 1, dim = 3)
distribution(y) <- multinomial(4, t(p), n_realisations = 5)
sample_distribution(p)

## error 2
alpha <- uniform(0, 10, dim = c(1, 5))
x <- dirichlet(alpha)
m <- model(x)
draws <- mcmc(m, n_samples = 100, warmup = 100, verbose = FALSE)

## error 3
n <- 10
k <- 3

# multinomial
size <- 5
x <- t(rmultinom(n, size, runif(k)))
p <- uniform(0, 1, dim = c(n, k))
distribution(x) <- multinomial(size, p)
m <- model(p)
expect_ok(draws <- mcmc(m, warmup = 0, n_samples = 5, verbose = FALSE))

## error 4
n <- 10
k <- 3

# categorical
x <- t(rmultinom(n, 1, runif(k)))
p <- uniform(0, 1, dim = c(n, k))
distribution(x) <- categorical(p)
m <- model(p)
expect_ok(draws <- mcmc(m, warmup = 0, n_samples = 5, verbose = FALSE))

## related to the above - test_joint.R ========================================
devtools::load_all(".")
source("tests/testthat/helpers.R")

## error 1 - Error: Could not find reasonable starting values after 20 attempts.
obs <- matrix(rbinom(300, 1, 0.5), 100, 3)
probs <- variable(0, 1, dim = 3)
distribution(obs) <- joint(
  bernoulli(probs[1]),
  bernoulli(probs[2]),
  bernoulli(probs[3]),
  dim = 100
)

sample_distribution(probs)

## error 2 - Error: all(above_lower & below_upper) is not TRUE
x <- joint(
  normal(0, 1, truncation = c(0, Inf)),
  normal(0, 2, truncation = c(-Inf, 0)),
  normal(-1, 1, truncation = c(1, 2))
)

sample_distribution(x, lower = c(0, -Inf, 1), upper = c(Inf, 0, 2))

## error 3 - Error: Could not find reasonable starting values after 20 attempts.
x <- joint(
  uniform(0, 1),
  uniform(0, 2),
  uniform(-1, 0)
)

sample_distribution(x, lower = c(0, 0, -1), upper = c(1, 2, 0))

# intriguingly, this also fails:

sample_distribution(uniform(0,1))
