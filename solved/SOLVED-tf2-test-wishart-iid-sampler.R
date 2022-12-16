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

  # also density


  # parameters to test
  m <- 5
  df <- m + 1
  sig <- rWishart(1, df, diag(m))[, , 1]

  # wrapper for argument names
  dwishart <- function(x, df, Sigma, log = FALSE) { # nolint
    ans <- MCMCpack::dwish(W = x, v = df, S = Sigma)
    if (log) {
      ans <- log(ans)
    }
    ans
  }

  # no vectorised wishart, so loop through all of these
  replicate(
    10,
    compare_distribution(
      greta::wishart,
      dwishart,
      parameters = list(
        df = df,
        Sigma = sig
      ),
      x = rWishart(1, df, sig)[, , 1],
      multivariate = TRUE
    )
  )
},
wd = "."
)
