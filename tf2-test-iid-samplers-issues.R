## test_iid_samples.R wishart distribution not sampling the same as rwish ------
reprex::reprex({
  devtools::load_all(".")
  source("tests/testthat/helpers.R")

  sigma <- rwish(1, 5, diag(4))[1, , ]
  prob <- t(runif(4))
  prob <- prob / sum(prob)

  compare_iid_samples(wishart,
                      rwish,
                      parameters = list(df = 7, Sigma = sigma)
  )

  # it looks like something is happening inside helpers, adjusting the wishard vector
  # before this code, it looks like this:
  # skimr::inline_hist(c(greta_samples))
  # [1] "▁▁▂▇▂▁▁▁"
  # skimr::inline_hist(c(r_samples))
  # [1] "▁▂▇▆▂▁▁▁"

  # this it does this
  # if it's a symmetric matrix, take only a triangle and flatten it
  # if (name %in% c("wishart", "lkj_correlation")) {
  #   include_diag <- name == "wishart"
  #   t_greta_samples <- apply(greta_samples, 1, get_upper_tri, include_diag)
  #   t_r_samples <- apply(r_samples, 1, get_upper_tri, include_diag)
  #   greta_samples <- t(t_greta_samples)
  #   r_samples <- t(t_r_samples)

  # after this, it looks like:
  # skimr::inline_hist(c(greta_samples))
  # [1] "▇▂▁▁▁▁▁▁"
  # skimr::inline_hist(c(r_samples))
  # [1] "▁▂▇▇▃▁▁▁"

},
wd = ".")

