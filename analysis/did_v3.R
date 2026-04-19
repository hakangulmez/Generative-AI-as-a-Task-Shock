# ================================================================
# DiD Analysis v3 — Two Frameworks × Two SME200 Definitions
# "Generative AI as a Task Shock"
# ================================================================
# Framework A: Eloundou task-based score (score_a)
# Framework B: C1-C10 rubric criteria score (score_b) [PRIMARY]
# SME200 M1:   avg annual revenue < $200M
# SME200 M2:   avg quarterly revenue < $50M (~$200M annual)
# ================================================================

library(fixest)
library(dplyr)

df <- read.csv("data/processed/master_panel.csv")
df$period_end  <- as.Date(df$period_end)

# Filter: 2020+, valid ln_revenue
df <- df[df$fiscal_year >= 2020 & !is.na(df$ln_revenue), ]

sig <- function(p) ifelse(p < 0.01, "***", ifelse(p < 0.05, "**", ifelse(p < 0.10, "*", "")))

wcb <- function(model, data, xvar, B = 999) {
  obs  <- coef(model)[xvar]
  yvar <- all.vars(formula(model))[1]
  data$y0 <- data[[yvar]] - obs * data[[xvar]]
  m0 <- feols(as.formula(paste("y0 ~", xvar, "| ticker + year_quarter")),
              data = data, cluster = ~ticker)
  res   <- residuals(m0)
  firms <- unique(data$ticker)
  set.seed(42)
  boot <- replicate(B, {
    w <- sample(c(-1, 1), length(firms), replace = TRUE)
    names(w) <- firms
    data$wt  <- w[data$ticker]
    data$yb  <- fitted(m0) + res * data$wt
    coef(feols(as.formula(paste("yb ~", xvar, "| ticker + year_quarter")),
               data = data, cluster = ~ticker))[xvar]
  })
  mean(abs(boot) >= abs(obs))
}

run_did <- function(data, score_col, label) {
  d <- data[!is.na(data[[score_col]]), ]
  if (n_distinct(d$ticker) < 10) {
    cat(sprintf("  %-40s SKIP (N<10)\n", label)); return(invisible(NULL))
  }
  mn   <- mean(d[[score_col]], na.rm = TRUE)
  sdev <- sd(d[[score_col]],   na.rm = TRUE)
  d$tx <- d$post * ((d[[score_col]] - mn) / sdev)
  m  <- feols(ln_revenue ~ tx | ticker + year_quarter, data = d, cluster = ~ticker)
  p  <- wcb(m, d, "tx")
  cat(sprintf("  %-40s N=%2d obs=%4d  beta=%7.4f  SE=%.4f  WCB_p=%.3f%s\n",
    label, n_distinct(d$ticker), nrow(d),
    coef(m)["tx"], se(m)["tx"], p, sig(p)))
  invisible(list(beta=coef(m)["tx"], se=se(m)["tx"], p=p, n=n_distinct(d$ticker)))
}

cat("================================================================\n")
cat("DiD v3: Two Frameworks × Two SME200 Definitions\n")
cat("================================================================\n\n")
cat(sprintf("Full panel: %d firms, %d obs\n\n", n_distinct(df$ticker), nrow(df)))

# ── 4 primary combinations ────────────────────────────────────────────────────
cat("=== PRIMARY: ln(Revenue) ~ Post × Score | firm FE + time FE ===\n\n")
cat(sprintf("  %-40s %4s %5s  %7s  %6s  %8s\n",
    "Specification", "N", "obs", "beta", "SE", "WCB_p"))
cat(strrep("-", 75), "\n")

# Minimum revenue filter: exclude nano-caps (avg quarterly < $5M ~ annual < $20M)
MIN_REV_M <- 5   # $5M quarterly minimum

sme_m1 <- df[df$sme_200_m1 == 1 & df$pre_rev_m >= MIN_REV_M, ]
sme_m2 <- df[df$sme_200_m2 == 1 & df$pre_rev_m >= MIN_REV_M, ]

results <- list()
for (sme_def in list(list(sme_m1, "M1 avg_annual<$200M"),
                     list(sme_m2, "M2 avg_qtrly<$50M"))) {
  cat(sprintf("\n--- SME200 %s (%d firms) ---\n", sme_def[[2]], n_distinct(sme_def[[1]]$ticker)))
  for (fw in list(list("score_b", "Framework B (C1-C10 Rubric)  [PRIMARY]"),
                  list("score_a", "Framework A (Eloundou Task)           "))) {
    lbl <- paste0(sme_def[[2]], " × ", fw[[2]])
    r <- run_did(sme_def[[1]], fw[[1]], lbl)
    results[[lbl]] <- r
  }
}

# ── Extended samples ──────────────────────────────────────────────────────────
cat("\n=== EXTENDED: SME500 & Full ===\n\n")
sme500 <- df[df$sme_500 == 1 & df$pre_rev_m >= MIN_REV_M, ]
for (fw in list(list("score_b", "Score B"), list("score_a", "Score A"))) {
  run_did(sme500, fw[[1]], paste0("SME500 × ", fw[[2]]))
  run_did(df,    fw[[1]], paste0("Full    × ", fw[[2]]))
}

# ── Mechanism: Table 3 for primary (B × M1) ──────────────────────────────────
cat("\n=== MECHANISM (Framework B × M1) ===\n\n")
cat(sprintf("  %-22s %8s %7s %8s %4s\n", "Outcome", "beta", "SE", "WCB_p", "N"))
cat(strrep("-", 54), "\n")

for (oc in list(
  list("ln_revenue",       "ln(Revenue)       "),
  list("gross_margin",     "Gross margin      "),
  list("operating_margin", "Operating margin  "),
  list("rd_intensity",     "R&D intensity     "),
  list("sga_intensity",    "SG&A intensity    ")
)) {
  d <- sme_m1[!is.na(sme_m1[[oc[[1]]]]) & !is.na(sme_m1$score_b), ]
  mn <- mean(d$score_b); sdev <- sd(d$score_b)
  d$tx <- d$post * ((d$score_b - mn) / sdev)
  tryCatch({
    m <- feols(as.formula(paste(oc[[1]], "~ tx | ticker + year_quarter")),
               data = d, cluster = ~ticker)
    p <- wcb(m, d, "tx")
    cat(sprintf("  %-22s %8.4f %7.4f %8.3f%s %3d\n",
        oc[[2]], coef(m)["tx"], se(m)["tx"], p, sig(p), n_distinct(d$ticker)))
  }, error = function(e) cat(sprintf("  %-22s  ERROR\n", oc[[2]])))
}

# ── Three-period (Framework B × M1) ──────────────────────────────────────────
cat("\n=== THREE-PERIOD (Framework B × M1) ===\n\n")
d3 <- sme_m1[!is.na(sme_m1$score_b), ]
mn3 <- mean(d3$score_b); sd3 <- sd(d3$score_b)
d3$lit_n   <- (d3$score_b - mn3) / sd3
d3$early   <- as.integer(d3$period_end >= as.Date("2022-10-01") &
                         d3$period_end <  as.Date("2024-01-01"))
d3$advanced <- as.integer(d3$period_end >= as.Date("2024-01-01"))
d3$tx_e    <- d3$early    * d3$lit_n
d3$tx_a    <- d3$advanced * d3$lit_n

m3 <- feols(ln_revenue ~ tx_e + tx_a | ticker + year_quarter, data = d3, cluster = ~ticker)
pe <- wcb(feols(ln_revenue ~ tx_e | ticker + year_quarter, data=d3, cluster=~ticker), d3, "tx_e")
pa <- wcb(feols(ln_revenue ~ tx_a | ticker + year_quarter, data=d3, cluster=~ticker), d3, "tx_a")

cat(sprintf("  Early AI  (2022Q4–2023Q4): beta=%7.4f  WCB_p=%.3f%s\n",
    coef(m3)["tx_e"], pe, sig(pe)))
cat(sprintf("  Advanced  (2024Q1+):       beta=%7.4f  WCB_p=%.3f%s\n",
    coef(m3)["tx_a"], pa, sig(pa)))
cat(sprintf("  Ratio Advanced/Early: %.2fx\n", coef(m3)["tx_a"] / coef(m3)["tx_e"]))

# ── Placebo ───────────────────────────────────────────────────────────────────
cat("\n=== PLACEBO (fake shock 2021Q2, Framework B × M1) ===\n\n")
pl <- sme_m1[sme_m1$fiscal_year <= 2022 & !is.na(sme_m1$score_b), ]
mn_pl <- mean(pl$score_b); sd_pl <- sd(pl$score_b)
pl$lit_n  <- (pl$score_b - mn_pl) / sd_pl
pl$post_pl <- as.integer(pl$fiscal_year == 2021 |
               (pl$fiscal_year == 2022 & pl$fiscal_quarter <= 3))
pl$tx_pl  <- pl$post_pl * pl$lit_n
mpl <- feols(ln_revenue ~ tx_pl | ticker + year_quarter, data = pl, cluster = ~ticker)
ppl <- wcb(mpl, pl, "tx_pl")
cat(sprintf("  Placebo (N=%d): beta=%.4f  WCB_p=%.3f%s  -> should be null\n",
    n_distinct(pl$ticker), coef(mpl)["tx_pl"], ppl, sig(ppl)))

cat("\n================================================================\n")
cat("DONE — did_v3.R\n")
cat("================================================================\n")
