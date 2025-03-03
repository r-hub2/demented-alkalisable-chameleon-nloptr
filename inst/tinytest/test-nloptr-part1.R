# Copyright (C) 2023 Avraham Adler. All Rights Reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
#
# File:   test-nloptr
# Author: Avraham Adler
# Date:   7 February 2023
#
# Test code in nloptr.R and nloptr.c which is not tested elsewhere.
#
# Changelog:
#   2023-08-23: Converted snapshots to testing portions of outputs and messages.
#
# It is possible for NLOPT to go slightly beyond maxtime or maxeval, especially
# for the global algorithms, which is why the stopping criterion has a
# weird-looking test. See
# https://nlopt.readthedocs.io/en/latest/NLopt_Reference/#stopping-criteria

library(nloptr)
options(digits = 7)

tol <- sqrt(.Machine$double.eps)

########################## Tests for nloptr.R ##################################

ctlNM <- list(algorithm = "NLOPT_LN_NELDERMEAD", xtol_rel = 1e-8,
              check_derivatives = TRUE)
ctlSQP <- list(algorithm = "NLOPT_LD_SLSQP", xtol_rel = 1e-8,
               check_derivatives = TRUE)

# internal function to check the arguments of the functions
expect_error(nloptr(3, "Zed"), "must be a function", fixed = TRUE)

fn <- function(x, b = NULL, c = NULL) x
expect_error(nloptr(3, fn, c = "Q"),
             "but this has not been passed to the 'nloptr' function",
             fixed = TRUE)

expect_error(nloptr(3, fn, b = "X", c = "Y", d = "Q"),
             "passed to (...) in 'nloptr' but this is not required in the",
             fixed = TRUE)

expect_warning(nloptr(3, fn, b = 3, c = 4, opts = ctlNM),
               "Skipping derivative checker because algorithm", fixed = TRUE)

########################## Tests for nloptr.c ##################################
ctl <- list(xtol_rel = 1e-8, maxeval = 1000L)
fn <- function(x) x ^ 2 - 4 * x + 4
lb <- 0
ub <- 6
optSol <- 2
optVal <- 0

## NLOPT_GN_DIRECT_L_NOSCAL
alg <- list(algorithm = "NLOPT_GN_DIRECT_L_NOSCAL")
testRun <- nloptr(5, fn, lb = lb, ub = ub, opts = c(alg, ctl))

expect_equal(testRun$solution, optSol, tolerance = tol)
expect_equal(testRun$objective, optVal, tolerance = tol)
expect_true(testRun$iterations <= ctl$maxeval + 5)
expect_true(testRun$status > 0)

## NLOPT_GN_DIRECT_L_RAND_NOSCAL
alg <- list(algorithm = "NLOPT_GN_DIRECT_L_RAND_NOSCAL")
testRun <- nloptr(5, fn, lb = lb, ub = ub, opts = c(alg, ctl))

expect_equal(testRun$solution, optSol, tolerance = tol)
expect_equal(testRun$objective, optVal, tolerance = tol)
expect_true(testRun$iterations <= ctl$maxeval + 5)
expect_true(testRun$status > 0)

## NLOPT_LN_PRAXIS
alg <- list(algorithm = "NLOPT_LN_PRAXIS")
testRun <- nloptr(5, fn, lb = lb, ub = ub, opts = c(alg, ctl))

expect_equal(testRun$solution, optSol, tolerance = tol)
expect_equal(testRun$objective, optVal, tolerance = tol)
expect_true(testRun$iterations <= ctl$maxeval + 5)
expect_true(testRun$status > 0)

## NLOPT_GN_MLSL
alg <- list(algorithm = "NLOPT_GN_MLSL")
lopts <- list(local_opts = list(algorithm = "NLOPT_LN_COBYLA", xtol_rel = 1e-8))
testRun <- nloptr(5, fn, lb = lb, ub = ub, opts = c(alg, ctl, lopts))

expect_equal(testRun$solution, optSol, tolerance = tol)
expect_equal(testRun$objective, optVal, tolerance = tol)
expect_true(testRun$iterations <= ctl$maxeval + 5)
expect_true(testRun$status > 0)

## NLOPT_GN_MLSL_LDS
alg <- list(algorithm = "NLOPT_GN_MLSL_LDS")
lopts <- list(local_opts = list(algorithm = "NLOPT_LN_COBYLA", xtol_rel = 1e-8))
testRun <- nloptr(5, fn, lb = lb, ub = ub, opts = c(alg, ctl, lopts))

expect_equal(testRun$solution, optSol, tolerance = tol)
expect_equal(testRun$objective, optVal, tolerance = tol)
expect_true(testRun$iterations <= ctl$maxeval + 5)
expect_true(testRun$status > 0)

## NLOPT_LN_AUGLAG_EQ
x0 <- c(-2, 2, 2, -1, -1)
fn1 <- function(x) exp(x[1] * x[2] * x[3] * x[4] * x[5])

eqn1 <- function(x) {
  c(x[1] * x[1] + x[2] * x[2] + x[3] * x[3] + x[4] * x[4] + x[5] * x[5],
    x[2] * x[3] - 5 * x[4] * x[5],
    x[1] * x[1] * x[1] + x[2] * x[2] * x[2])
}

optSol <- rep(0, 5)
optVal <- 1

testRun <- nloptr(x0, fn1, eval_g_eq = eqn1,
                  opts = list(algorithm = "NLOPT_LN_AUGLAG_EQ", xtol_rel = 1e-6,
                              maxeval = 10000L,
                              local_opts = list(algorithm = "NLOPT_LN_COBYLA",
                                                xtol_rel = 1e-6, maxeval = 1000L)))

expect_equal(testRun$solution, optSol, tolerance = tol)
expect_equal(testRun$objective, optVal, tolerance = tol)
expect_true(testRun$iterations <=  10005L)
expect_true(testRun$status > 0)

## NLOPT_LD_AUGLAG_EQ
gr1 <- function(x) {
  c(x[2] * x[3] * x[4] * x[5],
    x[1] * x[3] * x[4] * x[5],
    x[1] * x[2] * x[4] * x[5],
    x[1] * x[2] * x[3] * x[5],
    x[1] * x[2] * x[3] * x[4]) * exp(prod(x))
}

heqjac <- function(x) nl.jacobian(x0, eqn1)

testRun <- nloptr(x0, fn1, gr1, eval_g_eq = eqn1, eval_jac_g_eq = heqjac,
                  opts = list(algorithm = "NLOPT_LD_AUGLAG_EQ", xtol_rel = 1e-6,
                              maxeval = 10000L,
                              local_opts = list(algorithm = "NLOPT_LN_COBYLA",
                                                xtol_rel = 1e-6, maxeval = 1000L)))

expect_equal(testRun$solution, optSol, tolerance = tol)
expect_equal(testRun$objective, optVal, tolerance = tol)
expect_true(testRun$iterations <=  10005L)
expect_true(testRun$status > 0)
