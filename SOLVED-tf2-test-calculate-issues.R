# test_calculate: RNG seed isn't carried through ------------------------
devtools::load_all(".")
source("tests/testthat/helpers.R")

# fix variable
a <- normal(0, 1)
y <- normal(a, 1, dim = c(1, 3))

# the samples should be the same if the seed is the same
one <- calculate(y, nsim = 1, seed = 12345)
two <- calculate(y, nsim = 1, seed = 12345)
one
two
expect_identical(one, two)

## notes
# So I'm not even really sure where to start with this one
# I think I need to just step through calculate and look for seed steps?

debugonce(calculate)
calculate(y, nsim = 1, seed = 12345)

## so this bit is still there
#
## if an RNG seed was provided use it and reset the RNG on exiting
# if (!is.null(seed)) {
#   if (!exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
#     runif(1)
#   }
#
#   r_seed <- get(".Random.seed", envir = .GlobalEnv)
#   on.exit(assign(".Random.seed", r_seed, envir = .GlobalEnv))
#   set.seed(seed)
# }
# }
#
## However it seems that this isn't getting input later, or something?
## My only main thought on this is that we need to do the setting of the seed
## part a bit later, perhaps inside the TF function part,
## perhaps here?
    # # TF1/2
    # # need to wrap this in tf_function I think?
    # values <- calculate_target_tensor_list(
    #   dag,
    #   fixed_greta_arrays,
    #   values,
    #   stochastic,
    #   target,
    #   nsim
    # )
## As an aside, I think that function needs to be wrapped up in tf_function
##...
## OK so reading https://www.tensorflow.org/guide/random_numbers#stateless_rngs
## This looks like it might be really complicated?
## Might need to ask the TF folks from rstudio how they reckon we should manage this

