# test_operators.R ----
reprex::reprex({
devtools::load_all(".")
source("tests/testthat/helpers.R")

a <- randn(25, 4) > 0
b <- randn(25, 4) > 0
a[] <- as.integer(a[])
b[] <- as.integer(b[])

options(error = recover)
check_op(`!`, a, only = "data")
check_op(`&`, a, b, only = "data")
check_op(`|`, a, b, only = "data")
},
wd = ".")

# notes - this seems to be failing the log jacobian adjustment step somewhere
# the error is
# Error in py_call_impl(callable, dots$args, dots$keywords) :
#   RuntimeError: Evaluation error: TypeError: Expected int32, but got 0.0 of type 'float'.

# but it is confusing as
# `free` is
# Tensor("free_state_batch:0", shape=(1, 100), dtype=int32)
# Adn event_ndims is an integer:
# event_ndims <- as.integer(tf_bijector$forward_min_event_ndims)
# ljd <- tf_bijector$forward_log_det_jacobian(
#   x = free,
#   event_ndims = as.integer(event_ndims)
# )

# one thing I'm finding a bit challenging is that I can't seem to find a
# source of forward_log_det_jacobian

