# let's try and sample from a normal
# and get that working for TF2
# this feels like a relatively small goal, which should help me understand
# greta better and how it uses TF

library(greta)
devtools::load_all()
greta_sitrep()
x <- normal(0,1)
debug(calculate_list)
calculate(x, nsim = 10)

# navigating the debugger, I need to
  # first, step into `dag$define_tf()`
  # second, step into `self$define_tf_body(target_nodes = target_nodes)`
  # thirdly, step into:
     # ```
     # self$on_graph(
     #   lapply(target_nodes, function(x) x$define_tf(self))
     # )
     # ```
  # finally, this part of the code is where it fails
  # `with(self$tf_graph$as_default(), expr)`
  # It says:
  # Error in py_call_impl(callable, dots$args, dots$keywords) :
  #   TypeError: Expected int32, but got 44202948752.0 of type 'float'.

# then, on a fresh restart, if I try and disable various behaviours, I get:
library(greta)
greta_sitrep()
x <- normal(0,1)
devtools::load_all()
debug(calculate_list)
tf$compat$v1$disable_eager_execution()
tf$compat$v1$disable_v2_behavior()
calculate(x, nsim = 10)

# going through the same process of debugging as above...
# pretty much the same error
# Error in py_call_impl(callable, dots$args, dots$keywords) :
#   TypeError: Expected int32, but got 361944717931.0 of type 'float'.


# OK...so what if I start removing all the parts I think are to do with TF2?
# specifically, I've removed the `define_batch_size()` part, and passed
# batch_size as an argument. I set batch_size = NULL. This is for both of the `define_tf` functions that currently exist.
# I'm not sure how we can infer batch_size from the data, but as I
# understand it, batch_size was somehow passed from free_state?

devtools::load_all()
greta_sitrep()
x <- normal(0,1)
debug(calculate_list)
calculate(x, nsim = 10)

# OK! Slight progress, I now get this error:
# Error in py_call_impl(callable, dots$args, dots$keywords) :
#   RuntimeError: tf.placeholder() is not compatible with eager execution.

# So let's see if I can turn off eager execution?
# restart R
devtools::load_all()
tf$compat$v1$disable_eager_execution()
greta_sitrep()
x <- normal(0,1)
debug(calculate_list)
calculate(x, nsim = 10)

# OK! So now I get this error:
# Error in py_call_impl(callable, dots$args, dots$keywords) :
#   ValueError: Shape must be rank 3 but is rank 2 for '{{node Tile}} = Tile[T=DT_DOUBLE, Tmultiples=DT_INT32](Placeholder, Tile/multiples)' with input shapes: [1,1,1], [2].

# I've got NO IDEA what the third rank position is supposed to be here
# or for that matter, where that part comes from.
# on another note I seem to remember that this error might actually be the same
# if I turn on eager execution, and change
# `unbatched_tensor <- tf$compat$v1$placeholder(`
# in the `tf` function in `node_types.R` "data_node"
# to
# `unbatched_tensor <- tf$keras$Input(`

# OK.
# let's restart, and try again
devtools::load_all()
# tf$compat$v1$disable_eager_execution()
greta_sitrep()
x <- normal(0,1)
debug(calculate_list)
calculate(x, nsim = 10)

# OK so now the error is ... longer but more informative?

# Error in py_call_impl(callable, dots$args, dots$keywords) :
#   ValueError: Exception encountered when calling layer "tf.tile" (type TFOpLambda).
#
# Shape must be rank 4 but is rank 2 for '{{node tf.tile/Tile}} = Tile[T=DT_DOUBLE, Tmultiples=DT_INT32](Placeholder, tf.tile/Tile/multiples)' with input shapes: [?,1,1,1], [2].
#
# Call arguments received:
#   • input=tf.Tensor(shape=(None, 1, 1, 1), dtype=float64)
# • multiples=['1', '1']
# • name=None

# my main thought with this is that it's quite possible I'm still getting the
# interface wrong for TF 2...or I need to look deeper into the code.

# now let's rerun it with a `browser()` in define_tf() of `node_class.R`...
# ugh
# restart R
devtools::load_all()
greta_sitrep()
x <- normal(0,1)
calculate(x, nsim = 10)

# A-HAH! It happens at a line with tf$tile ...let's pop a browser in there
# which happens in the tf = function() part of "data_node"
# restart...rinse and repeat.
devtools::load_all()
greta_sitrep()
x <- normal(0,1)
calculate(x, nsim = 10)

#OK, so I think this is happening because `batch_size` is currently not being defined.
# let's define it and throw it back in, in the self$define_batch_size() part
# although I think that this will not work since it uses tf_run, and placeholder.
# let's see.
# rinse, repeat.
devtools::load_all()
tf$compat$v1$disable_eager_execution()
greta_sitrep()
x <- normal(0,1)
calculate(x, nsim = 10)

# OK - so same error. Which is...strange, but still, it is something to do with
# tf.tile.
# if I enable eager execution, then I get some further details...
# - perhaps the arguments all need to be tensors and they aren't currently?
# but if I disable eager execution, I get this error:

# Error in py_call_impl(callable, dots$args, dots$keywords) :
#   ValueError: Tensor("Placeholder:0", dtype=int32) must be from the same graph as Tensor("Placeholder:0", shape=(1, 1, 1), dtype=float64) (graphs are <tensorflow.python.framework.ops.Graph object at 0x1713ce550> and <tensorflow.python.framework.ops.Graph object at 0x17952f460>).

# I wonder if I can disable all behaviour, and if this changes things?
devtools::load_all()
tf$compat$v1$disable_eager_execution()
tf$compat$v1$disable_v2_behavior()
greta_sitrep()
x <- normal(0,1)
calculate(x, nsim = 10)

# nope.
# Error in py_call_impl(callable, dots$args, dots$keywords) :
# ValueError: Tensor("Placeholder:0", dtype=int32) must be from the same graph as Tensor("Placeholder:0", shape=(1, 1, 1), dtype=float64) (graphs are <tensorflow.python.framework.ops.Graph object at 0x16487e550> and <tensorflow.python.framework.ops.Graph object at 0x1718ae400>).

# OK, so now let's think back to the idea of creating lots of functions here instead of placeholders and stuff...

# which brings me to - what is a batch size?
# is the batch size just the number of values to sample from the normal?
# where even is that detail passed?
# answer: batch size in this case is basically the number of samples
# that we are drawing. So the sample size in sample().
  # we then pass this to the feed dict
  # however...I'm unsure what this really means in the greater context
  # as we still get batch_size defined when you do model(normal(0,1))
  # and so I'm not sure that to think about that
# previously, this required that you passed an appropriately sized
# tensor of the right size
# but now we can just literally pass the parameters along
# so we can most likely get rid of all of the calls to placeholder
# and a lot of the technology around managing that.
# this means that we will need to do a bit of a conscious uncoupling
# of many of the handlers around batch size and all this stuff
# which will overall result in simpler code
# but it feels like it will be messy before we get there.

# ---- mcmc? ----
load_all()
greta_sitrep()
mcmc(model(normal(0,1)), n_samples = 10)

    # Error in py_call_impl(callable, dots$args, dots$keywords) :
    #   TypeError: <tf.Tensor 'Placeholder:0' shape=(None, 1) dtype=float64> is out of scope and cannot be used here. Use return values, explicit Python locals or TensorFlow collections to access it.
    # <... omitted ...>ibrary.py", line 740, in _apply_op_helper
    #       op = g._create_op_internal(op_type_name, inputs, dtypes=None,
    #     File "/Users/nick/Library/r-miniconda-arm64/envs/greta-env/lib/python3.8/site-packages/tensorflow/python/framework/ops.py", line 3776, in _create_op_internal
    #       ret = Operation(
    #     File "/Users/nick/Library/r-miniconda-arm64/envs/greta-env/lib/python3.8/site-packages/tensorflow/python/framework/ops.py", line 2175, in __init__
    #       self._traceback = tf_stack.extract_stack_for_node(self._c_op)
    #
    # The tensor <tf.Tensor 'Placeholder:0' shape=(None, 1) dtype=float64> cannot be accessed from here, because it was defined in <tensorflow.python.framework.ops.Graph object at 0x280e18100>, which is out of scope.
    # See `reticulate::py_last_error()` for details

# OK, let's delt deeper
debug(mcmc)
mcmc(model(normal(0,1)), n_samples = 10)

# ooof, OK so the error happens here:
    # target_greta_arrays <- model$target_greta_arrays
# even just printing the `model` object causes the same error
# so let's look into `model`
load_all()
greta_sitrep()
debug(model)
model(normal(0,1))

# hooboy, finally struck gold. It's all in the tf_log_jacobian_adjustment
# which is what Nick G said in the first place, but it's had to become
# clear for me
# so we drop a browser() in tf_log_jacobian_adjustment, and away we go
# restart R
load_all()
greta_sitrep()
# debug(model)
model(normal(0,1))

# ----- debugging 2022-06-14
devtools::load_all()
x <- normal(0,1)
mcmc(model(x))


# ---- debugging 2022-06-17
# browser set on line 705 of inference_class.R
# which is just before we run  sampler_batch <- tfp$mcmc$sample_chain(
devtools::load_all()
x <- normal(0, 1)
m <- model(x)
# debug(mcmc)
mcmc(m)

# hit n to take one step from browser
# now try and run GradientTape on the inputs
# load tensorflow to get access to `%as` method
library(tensorflow)
# Attaching package: ‘tensorflow’
#
# The following object is masked from ‘package:greta’:
#
#   tf
x <- as_tensor(free_state)
x
# tf.Tensor(
#   [[-0.06387527]
#    [-0.03749587]
#    [ 0.0033471 ]
#    [ 0.07576335]], shape=(4, 1), dtype=float64)
with(
  tf$GradientTape() %as% tape, {
    browser()
    tape$watch(x)
    y <- self$model$dag$tf_log_prob_function_adjusted(x)
  }
)
# debug at #3: tape$watch(x)
dy_dx <- tape$gradient(y, x)
dy_dx
# NULL

#---- debug 2022-06-22

library(tensorflow)
library(greta)
debugonce(normal)
normal(0,1)

foo <- function() {
  x <- tf$ones(shape(2, 2))

  with(tf$GradientTape() %as% t, {
    t$watch(x)
    y <- tf$reduce_sum(x)
    z <- tf$multiply(y, y)
  })

  # Derivative of z with respect to the original input tensor x
  dz_dx <- t$gradient(z, x)
  dz_dx
}
debugonce(foo)
foo()


### restart
# trying unadjusted
# trying adjustment
# trying density
devtools::load_all()
x <- normal(0, 1)
m <- model(x)
draws <- mcmc(m)
summary(draws)
library(tensorflow)

explore_grad <- function(){
  x <- as_tensor(free_state)
  x
  with(
    tf$GradientTape(persistent = TRUE) %as% tape, {
      tape$watch(x)
      y_adjusted <- self$model$dag$tf_log_prob_function_adjusted(x)
      y_adjustment <- self$model$dag$tf_log_prob_function_adjustment(x)
      y_density <- self$model$dag$tf_log_prob_function_density(x)
      y_unadjusted <- self$model$dag$tf_log_prob_function_unadjusted(x)
      y_free <- self$model$dag$tf_log_prob_function_variable_1_free(x)
      y_1 <- self$model$dag$tf_log_prob_function_variable_1(x)

    }
  )
  # debug at #3: tape$watch(x)
  dy_adjusted_dx <- tape$gradient(y_adjusted, x)
  dy_adjustment_dx <- tape$gradient(y_adjustment, x)
  dy_density_dx <- tape$gradient(y_density, x)
  dy_unadjusted_dx <- tape$gradient(y_unadjusted, x)
  dy_free_dx <- tape$gradient(y_free, x)
  dy_1_dx <- tape$gradient(y_1, x)
  dy_adjusted_dx
  dy_adjustment_dx
  dy_density_dx
  dy_unadjusted_dx
}

debugonce(explore_grad)
explore_grad()

# debugger ---- 2022-06-22

devtools::load_all()
x <- normal(0, 1)
m <- model(x)
mcmc(m)
library(tictoc)

tic()
draws <- mcmc(m, warmup = 1, n_samples = 1000, pb_update = 1)
toc()

tic()
draws <- mcmc(m, warmup = 1, n_samples = 1000, pb_update = 50)
toc()

tic()
draws <- mcmc(m, warmup = 1, n_samples = 1000, pb_update = 250)
toc()

tic()
draws <- mcmc(m, warmup = 1, n_samples = 1000, pb_update = 500)
toc()


# ---- degbug 2022-06-23 :: changing input_signature in tf_function
devtools::load_all()
x <- normal(0, 1)
m <- model(x)
mcmc(m)

tf$reshape(
  tensorflow::as_tensor(
    hmc_epsilon * (hmc_diag_sd / tf$reduce_sum(hmc_diag_sd))
    ),
  # TF1/2 - what do we do with the free_state here?
  shape = shape(as.integer(free_state_size))
)

# 2022-07-13 ----
reprex::reprex({
  devtools::load_all(".")
  x <- normal(0, 1)
  m <- model(x)
  library(tictoc)
  tic()
  draws <- mcmc(m, n_samples = 500, warmup = 500)
  toc()
  library(coda)
  plot(draws)
},
wd = ".")

# after adding the input_signature stuff:
reprex::reprex({
  devtools::load_all(".")
  x <- normal(0, 1)
  m <- model(x)
  draws <- mcmc(m, n_samples = 500, warmup = 500)
},
wd = ".")

# 2022-07-26 debug using browser to check size/shape of hmc_* input ----
reprex::reprex({
library(tictoc)
devtools::load_all(".")
x <- normal(0, 1)
m <- model(x)
tic()
draws <- mcmc(m, n_samples = 500, warmup = 500)
toc()
library(coda)
plot(draws)
},
wd = ".")

# 2022-07-26 investigating the tuning parameters
reprex::reprex({
  library(tictoc)
  devtools::load_all(".")
  x <- normal(0, 1)
  m <- model(x)
  tic()
  draws <- mcmc(m, n_samples = 500, warmup = 500)
  toc()
  library(coda)
  plot(draws)
},
wd = ".")


# compare CRAN greta TF1
# compare new greta TF2
# for each of these, benchmark the time it takes to run the log_prob function
# on a big and a small array

# TF1 code to try out and profile
x <- normal(0, 1)
m <- model(x)
tic()
draws <- mcmc(m, n_samples = 500, warmup = 500)
glp_fun <- m$dag$generate_log_prob_function()
glp_fun(array(1, c(1, 1)))

# we can profile the code on this - TF2 code
build_array <- function(n){
  array(n, c(n,1))
}

bm <- bench::mark(
  x_0 = glp_fun(build_array(1)),
  x_3 = glp_fun(build_array(1e3)),
  x_6 = glp_fun(build_array(1e6)),
  x_7 = glp_fun(build_array(1e7)),
  check = FALSE
)

plot(bm)

# if there's a big difference between TF1 and TF2
# then see if there's something to do with TF code that we don't understand

plot(bm)

plot(draws)

### 2022-08-05
devtools::load_all(".")

fun <- function(free_state) {
  norm <- tfp$distributions$Normal(0, 1)
  norm$log_prob(free_state)
}

hmc_model <- tfp$mcmc$HamiltonianMonteCarlo(
  target_log_prob_fn = fun,
  step_size = 1L,
  num_leapfrog_steps = 4L
)

hmc_model

tfautograph::view_function_graph(
  fn = tfp$mcmc$HamiltonianMonteCarlo,
  args = list(
    target_log_prob_fn = fun,
    step_size = 1L,
    num_leapfrog_steps = 4L
  ),
  profiler = TRUE
)

# womp womp

# but this works:

tfp$mcmc$sample_chain(
  num_results = 10L,
  current_state = 1,
  kernel = hmc_model,
  trace_fn = function(current_state, kernel_results) {
    kernel_results
  },
  num_burnin_steps = tf$constant(0L, dtype = tf$int32),
  num_steps_between_results = 0L,
  parallel_iterations = 1L
)

# does this work?
tfautograph::view_function_graph(
  fn = tfp$mcmc$sample_chain,
  args = list(
    num_results = 10L,
    current_state = 1,
    kernel = hmc_model,
    trace_fn = function(current_state, kernel_results) {
      kernel_results
    },
    num_burnin_steps = tf$constant(0L, dtype = tf$int32),
    num_steps_between_results = 0L,
    parallel_iterations = 1L
  ),
  profiler = TRUE
  )

# nope

# OK - but if we turn on the browser() just before
## result <- cleanly(self$tf_evaluate_sample_batch(
# then we can do:
# restart R
devtools::load_all(".")
x <- normal(0,1)
m <- model(x)
draws <- mcmc(m)

fn <- self$tf_evaluate_sample_batch

with(parent.frame(),fn(
  free_state = tensorflow::as_tensor(free_state,
                                     dtype = tf_float()),
  sampler_burst_length = tensorflow::as_tensor(sampler_burst_length),
  sampler_thin = tensorflow::as_tensor(sampler_thin),
  sampler_param_vec = tensorflow::as_tensor(sampler_param_vec,
                                            dtype = tf_float(),
                                            shape = length(sampler_param_vec))
))

tfautograph::view_function_graph(
  fn = self$tf_evaluate_sample_batch,
  args = list(
    free_state = tensorflow::as_tensor(free_state,
                                       dtype = tf_float()),
    sampler_burst_length = tensorflow::as_tensor(sampler_burst_length),
    sampler_thin = tensorflow::as_tensor(sampler_thin),
    sampler_param_vec = tensorflow::as_tensor(sampler_param_vec,
                                              dtype = tf_float(),
                                              shape = length(sampler_param_vec))
  ),
  profiler = TRUE
)


result <- self$tf_evaluate_sample_batch(
  free_state = tensorflow::as_tensor(free_state,
                                     dtype = tf_float()),
  sampler_burst_length = tensorflow::as_tensor(sampler_burst_length),
  sampler_thin = tensorflow::as_tensor(sampler_thin),
  sampler_param_vec = tensorflow::as_tensor(sampler_param_vec,
                                            dtype = tf_float(),
                                            shape = length(sampler_param_vec))
)

tfautograph::view_function_graph(
  fn = self$tf_evaluate_sample_batch,
  args = list(
    free_state = tensorflow::as_tensor(free_state,
                                       dtype = tf_float()),
    sampler_burst_length = tensorflow::as_tensor(sampler_burst_length),
    sampler_thin = tensorflow::as_tensor(sampler_thin),
    sampler_param_vec = tensorflow::as_tensor(sampler_param_vec,
                                              dtype = tf_float(),
                                              shape = length(sampler_param_vec))
  ),
  profiler = TRUE
)

### try another approach with using the

# function (fn, args, ..., name = deparse(substitute(fn)), profiler = FALSE,
#           concrete_fn = do.call(fn$get_concrete_fn, args), graph = concrete_fn$graph)
# {
devtools::load_all(".")
x <- normal(0,1)
m <- model(x)
draws <- mcmc(m)

# logdir <- glue::glue("../log{lubridate::today()}")
# fs::dir_create(logdir)
logdir <- tempfile(pattern = "tflogdir")

writer <- tf$summary$create_file_writer(logdir)
tf$compat$v2$summary$trace_on(graph = TRUE, profiler = TRUE)

# result <- cleanly(self$tf_evaluate_sample_batch(
self$tf_evaluate_sample_batch(
  free_state = tensorflow::as_tensor(free_state,
                                     dtype = tf_float()),
  sampler_burst_length = tensorflow::as_tensor(sampler_burst_length),
  sampler_thin = tensorflow::as_tensor(sampler_thin),
  sampler_param_vec = tensorflow::as_tensor(sampler_param_vec,
                                            dtype = tf_float(),
                                            shape = length(sampler_param_vec))
)
##

with(writer$as_default(), {
      tf$summary$trace_export(name = "self$tf_evaluate_sample_batch",
                              step = 0L,
                              profiler_outdir = logdir)
    })
tf$summary$trace_off()
  # tensorboard <- get("tensorboard", envir = asNamespace("tensorflow"))
tensorflow::tensorboard(log_dir = logdir, reload_interval = 0L)
tf$summary$trace_off()
###


##### another attempt
devtools::load_all(".")

log_directory <- glue::glue("../log{lubridate::today()}")
fs::dir_create(log_directory)

tb_callback <- tensorflow::tensorboard(log_directory)

tb_callback

fun <- function(free_state) {
  norm <- tfp$distributions$Normal(0, 1)
  norm$log_prob(free_state)
}

hmc_model <- tfp$mcmc$HamiltonianMonteCarlo(
  target_log_prob_fn = fun,
  step_size = 1L,
  num_leapfrog_steps = 4L
)

###
tfautograph::view_function_graph(
  fn = tfp$mcmc$HamiltonianMonteCarlo,
  args = list(
    target_log_prob_fn = fun,
    step_size = 1L,
    num_leapfrog_steps = 4L
  ),
  profiler = TRUE
)



some_draws <- tfp$mcmc$sample_chain(
  num_results = 10L,
  current_state = 1,
  kernel = hmc_model,
  trace_fn = function(current_state, kernel_results) {
    kernel_results
  },
  num_burnin_steps = tf$constant(0L, dtype = tf$int32),
  num_steps_between_results = 0L,
  parallel_iterations = 1L
)

some_draws

fun <- function(free_state) {
  norm <- tfp$distributions$Normal(0, 1)
  norm$log_prob(free_state)
}

hmc_model <- tfp$mcmc$HamiltonianMonteCarlo(
  target_log_prob_fn = fun,
  step_size = 1L,
  num_leapfrog_steps = 4L
)

my_define_tf_draws <- function(){
  tfp$mcmc$sample_chain(
    num_results = 10L,
    current_state = 1,
    kernel = hmc_model,
    trace_fn = function(current_state, kernel_results) {
      kernel_results
    },
    num_burnin_steps = tf$constant(0L, dtype = tf$int32),
    num_steps_between_results = 0L,
    parallel_iterations = 1L
  )
}

my_tf_hmc <- tensorflow::tf_function(
  f = my_define_tf_draws,
  input_signature = list(
    # free state
    tf$TensorSpec(shape = list(NULL, 4L),
                  dtype = tf_float()),
    # sampler_burst_length
    tf$TensorSpec(shape = list(),
                  dtype = tf$int32),
    # sampler_thin
    tf$TensorSpec(shape = list(),
                  dtype = tf$int32),
    # sampler_param_vec
    tf$TensorSpec(shape = list(4L),
                  dtype = tf_float())
  )
)


# 2022-08-05
# trying a pared down approach

library(tfautograph)
load_all(".")

fun <- function(free_state) {
  norm <- tfp$distributions$Normal(0, 1)
  norm$log_prob(free_state)
}

hmc_model <- tfp$mcmc$HamiltonianMonteCarlo(
  target_log_prob_fn = fun,
  step_size = 1L,
  num_leapfrog_steps = 4L
)

my_define_tf_draws <- tensorflow::tf_function(function(){
  tfp$mcmc$sample_chain(
    num_results = 10L,
    current_state = 1,
    kernel = hmc_model,
    trace_fn = function(current_state, kernel_results) {
      kernel_results
    },
    num_burnin_steps = tf$constant(0L, dtype = tf$int32),
    num_steps_between_results = 0L,
    parallel_iterations = 1L
  )
}
)

logdir <- tempfile(pattern = "tflogdir")
writer <- tf$summary$create_file_writer(logdir)
tf$compat$v2$summary$trace_on(graph = TRUE, profiler = TRUE)

my_define_tf_draws()
my_define_tf_draws()
my_define_tf_draws()
my_define_tf_draws()

# do.call(fn, args)
with(writer$as_default(), {
  tf$summary$trace_export(name = "whatever", step = 0L, profiler_outdir = logdir)
})
tensorboard <- get("tensorboard", envir = asNamespace("tensorflow"))
tensorboard(log_dir = logdir, reload_interval = 0L)

tfautograph::view_function_graph(fn, list(tf$constant(5)))



### but with greta

# 2022-08-05
# trying a pared down approach

library(tfautograph)
load_all(".")

logdir <- tempfile(pattern = "tflogdir")
writer <- tf$summary$create_file_writer(logdir)
tf$compat$v2$summary$trace_on(graph = TRUE, profiler = TRUE)


###
# but with greta
x <- normal(0,1)
m <- model(x)
draws <- mcmc(m, n_samples = 100, warmup = 100)
###

# do.call(fn, args)
with(writer$as_default(), {
  tf$summary$trace_export(name = "whatever", step = 0L, profiler_outdir = logdir)
})
tensorboard <- get("tensorboard", envir = asNamespace("tensorflow"))
tensorboard(log_dir = logdir, reload_interval = 0L)

tfautograph::view_function_graph(fn, list(tf$constant(5)))


### what if we set the CPU on instead?

library(greta)
greta_sitrep()

x <- normal(0,1)
m <- model(x)
draws_cpu <- mcmc(m, n_samples = 50, warmup = 50, compute_options = cpu_only())
draws_gpu <- mcmc(m, n_samples = 50, warmup = 50, compute_options = gpu_only())

library(coda)
plot(draws_cpu)
plot(draws_gpu)

## trying to get `calculate` to work with TF2
library(greta)
greta_sitrep()
x <- normal(0, 1, dim = 3)
a <- lognormal(0, 1)
y <- sum(x^2) + a
calculate(y,
          values = list(x = c(0.1, 0.2, 0.3), a = 2))

# ok now try with something more complex
alpha <- normal(0, 1)
beta <- normal(0, 1)
sigma <- lognormal(1, 0.1)
y <- as_data(iris$Petal.Width)
mu <- alpha + iris$Petal.Length * beta
distribution(y) <- normal(mu, sigma)
m <- model(alpha, beta, sigma)

# sample values of the parameters, or different observation data (y), from
# the priors (useful for prior # predictive checking)
calculate(alpha, beta, sigma, nsim = 10)

x <- normal(0, 1)
m <- model(x)
calculate(x, nsim = 10)
# OK...rad? The sampling mode stuff is broken

# OK now let's go back to something more complex
# ok now try with something more complex
# define a model
alpha <- normal(0, 1)
beta <- normal(0, 1)
sigma <- lognormal(1, 0.1)
y <- as_data(iris$Petal.Width)
mu <- alpha + iris$Petal.Length * beta
distribution(y) <- normal(mu, sigma)
m <- model(alpha, beta, sigma)

# sample values of the parameters, or different observation data (y), from
# the priors (useful for prior # predictive checking) - see also
# ?simulate.greta_model
# debugonce(calculate_target_tensor_list)
calculate(alpha, beta, sigma, nsim = 100)
# SICK!

# OK now let's use calculate with MCMC
#####
# the values are this kind of tensorflow ish thing
# and they should probably be just values - like the following example.
x <- normal(0, 1)
calculate(x, nsim = 5)

# if the greta array only depends on data,
# you can pass an empty list to values (this is the default)
x <- ones(3, 3)
y <- sum(x)
calculate(y)

# a simple model works
# devtools::load_all(".")
library(greta)
greta_sitrep()
ln <- lognormal(0,1)
m_ln <- model(ln)
draws_ln <- mcmc(m_ln, n_samples = 500, warmup = 500)

library(coda)
plot(draws_ln)
# hmmm!

library(greta)
# define a simple Bayesian model
x <- rnorm(10)
mu <- normal(0, 5)
sigma <- lognormal(1, 0.1)
distribution(x) <- normal(mu, sigma)
m <- model(mu, sigma)

# carry out mcmc on the model
draws <- mcmc(m, n_samples = 100)

plot(draws)

# try out calculate with mcmc
devtools::load_all(".")
greta_sitrep()
samples <- 10
x <- as_data(c(1, 2))
a <- normal(0, 1)
y <- a * x
m <- model(y)
draws <- mcmc(m, warmup = 0, n_samples = samples, verbose = FALSE)

# with an existing greta array
debugonce(calculate_greta_mcmc_list)
y_values <- calculate(y, values = draws)

y_values_sims <- calculate(y, values = draws, nsim = 10)

# stochastic simulations
samples <- 10
chains <- 2
n <- 100
y <- as_data(rnorm(n))
x <- as_data(1)
a <- normal(0, 1)
distribution(y) <- normal(a, x)
m <- model(a)
draws <- mcmc(
  m,
  warmup = 0,
  n_samples = samples,
  chains = chains,
  verbose = FALSE
)

# new stochastic greta array
b <- lognormal(a, 1)

sims <- calculate(b, values = draws, nsim = 10)

###
# currently a lot of the helper functions in greta need to be rewritten as they
# use various TF1 things that break a lot of existing code
# and as far as I can tell, there are issues with how some of the code
# is initialising TF with dag$define_tf()
# - so this is happening and then the free state doesn't
# exist...and I'm not sure how to resolve this
# current tests that fail:
  # (2 errors) test_variables fails due to some error regarding
    # Some error regarding sampling of truncated variables...
###
###
devtools::load_all(".")
source("tests/testthat/helpers.R")

# test_distributions is failing

# 2022-10-12 tests that fail ---------------------------------------------------

# test_operators
# test_iid_samples
# test_opt ***new*** - written issue for
# test_inference ***new*** - writing an issue...
# test_misc is mucking up due to version mocking ***new***
