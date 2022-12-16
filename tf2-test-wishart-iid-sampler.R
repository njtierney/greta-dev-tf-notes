reprex::reprex({
  devtools::load_all("../greta/")
  source("../greta/tests/testthat/helpers.R")

  sigma <- rwish(1, 5, diag(4))[1, , ]
  prob <- t(runif(4))
  prob <- prob / sum(prob)

  g_wish <- wishart(
    df = 7,
    Sigma = sigma
  )

  greta_samples <- calculate(g_wish, nsim = 3)
  greta_samples$g_wish

  r_wish <- rwish(
    n = 3,
    df = 6,
    Sigma = sigma
  )

  r_wish

  # regular R wishart
  rWishart(n = 3,
           df = 7,
           Sigma = sigma)

  compare_iid_samples(wishart,
                      rwish,
                      parameters = list(df = 7, Sigma = sigma)
  )
},
wd = "."
)
