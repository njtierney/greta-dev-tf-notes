## SOLVED

# test_function.R fails --------------------
devtools::load_all(".")
source("tests/testthat/helpers.R")

k <- 4
x <- rWishart(1, k + 1, diag(k))[, , 1]
x_ga <- as_data(x)

r_out <- eigen(x)
greta_out <- eigen(as_data(x))

r_out_vals <- eigen(x, only.values = TRUE)
greta_out_vals <- eigen(as_data(x), only.values = TRUE)

# # values
# compare_op(r_out$values, grab(greta_out$values))
#
# # only values
# compare_op(r_out_vals$values, grab(greta_out_vals$values))

# first issue is with `grab`
grab(greta_out$values)
options(error = recover)
grab(greta_out$values)

###
# 1: grab(greta_out$values)
# 2: helpers.R#32: node$define_tf(dag)
# 3: node_class.R#195: self$tf(dag)
# 4: node_types.R#183: do.call(operation, tf_args)
# 5: (function (x)
# {
#   vals <- x[[1]]
#   dim <- tf$constant(1, shape = list(1))
#
#   6: tf_functions.R#549: tf$constant(1, shape = list(1))

# so dropping into the `do.call` bit
# OK so it looks like maybe it was just
# tf_functions.R#549: tf$constant(1, shape = list(1))
# needed to be
# tf_functions.R#549: tf$constant(1, shape = list(1L))
###
grab(greta_out$values)

## OK now this fails
# compare_op(r_out_vals$values, grab(greta_out_vals$values))
options(error = recover)

compare_op(r_out_vals$values, grab(greta_out_vals$values))

###
# Error in py_call_impl(callable, dots$args, dots$keywords) :
#   TypeError: Dimension value must be integer or None or have an __index__
#   method, got value '1.0' with type '<class 'float'>'
# Enter a frame number, or 0 to exit
#
# 1: compare_op(r_out_vals$values, grab(greta_out_vals$values))
# 2: testthat-helpers.R#45: as.vector(abs(r_out - greta_out))
# 3: grab(greta_out_vals$values)
# 4: helpers.R#32: node$define_tf(dag)
# 5: node_class.R#195: self$tf(dag)
# 6: node_types.R#183: do.call(operation, tf_args)
# 7: (function (x)
# {
#   vals <- tf$linalg$eigvalsh(x)
#   dim <- tf$constant(1, s
#                      8: tf_functions.R#537: tf$constant(1, shape = list(1))
#                      9: py_call_impl(callable, dots$args, dots$keywords)
###

## OK so main fix was ensuring the `1` was `1L` in the following:
#

reprex::reprex({
devtools::load_all(".")
source("tests/testthat/helpers.R")

  k <- 4
  x <- rWishart(1, k + 1, diag(k))[, , 1]
  x_ga <- as_data(x)

  r_out <- eigen(x)
  greta_out <- eigen(as_data(x))

  r_out_vals <- eigen(x, only.values = TRUE)
  greta_out_vals <- eigen(as_data(x), only.values = TRUE)

  # values
  compare_op(r_out$values, grab(greta_out$values))

  # only values
  compare_op(r_out_vals$values, grab(greta_out_vals$values))
},
wd = ".")
