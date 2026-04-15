# ================================================
# Main DiD Analysis — Pipeline v2
# "Generative AI as a Task Shock"
# ================================================
# PRIMARY: Literature Rubric (continuous, SME <$200M)
# EXTENDED: SME <$500M, Full sample
# THREE-PERIOD: Early AI / Advanced AI
# QUARTILE: Q75 vs Q25
# ================================================

library(fixest)
library(dplyr)

df <- read.csv("data/processed/master_panel_v2.csv")
df$period_end <- as.Date(df$period_end)
df$year_quarter <- paste0(format(df$period_end, "%Y"), "Q",
                          ceiling(as.numeric(format(df$period_end, "%m")) / 3))
df <- df[!is.na(df$ln_revenue) & !is.na(df$rho), ]

# Normalize rho to [0,1]
df$rho_n <- df$rho / 100

# Samples (pre-filtered in panel)
sme200 <- df[df$sme_200 == 1, ]
sme500 <- df[df$sme_500 == 1, ]
full   <- df[df$full_sample == 1, ]

cat("SME $200M:", n_distinct(sme200$ticker), "firms\n")
cat("SME $500M:", n_distinct(sme500$ticker), "firms\n")
cat("Full:     ", n_distinct(full$ticker), "firms\n\n")

# ---- WCB FUNCTION ----
# Wild cluster bootstrap (null imposed, Rademacher weights)
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
    data$wt <- w[data$ticker]
    data$yb <- fitted(m0) + res * data$wt
    coef(feols(as.formula(paste("yb ~", xvar, "| ticker + year_quarter")),
               data = data, cluster = ~ticker))[xvar]
  })
  mean(abs(boot) >= abs(obs))
}

sig <- function(p) ifelse(p < 0.01, "***", ifelse(p < 0.05, "**", ifelse(p < 0.10, "*", "")))

# ============================================
# TABLE 1: PRIMARY RESULTS
# ============================================
cat("=== TABLE 1: PRIMARY RESULTS ===\n\n")

for (d in list(list(sme200, "SME $200M"), list(sme500, "SME $500M"), list(full, "Full"))) {
  dat <- d[[1]]; label <- d[[2]]
  dat$tx <- dat$post * dat$rho_n
  m <- feols(ln_revenue ~ tx | ticker + year_quarter, data = dat, cluster = ~ticker)
  p <- wcb(m, dat, "tx")
  cat(sprintf("%s (N=%d): beta=%.4f SE=%.4f WCB=%.3f%s\n",
    label, n_distinct(dat$ticker), coef(m)["tx"], se(m)["tx"], p, sig(p)))
}

# ============================================
# PLACEBO (pre-period only, fake shock 2021Q1)
# ============================================
cat("\n=== PLACEBO (pre-period) ===\n")
pl <- sme200[sme200$period_end < as.Date("2022-10-01"), ]
pl$post_pl <- as.integer(pl$period_end >= as.Date("2021-01-01"))
pl$tx_pl <- pl$post_pl * pl$rho_n
mpl <- feols(ln_revenue ~ tx_pl | ticker + year_quarter, data = pl, cluster = ~ticker)
ppl <- wcb(mpl, pl, "tx_pl")
cat(sprintf("Placebo: beta=%.4f WCB=%.3f%s\n", coef(mpl)["tx_pl"], ppl, sig(ppl)))

# ============================================
# THREE-PERIOD ANALYSIS (Early / Advanced AI)
# ============================================
cat("\n=== THREE-PERIOD ANALYSIS ===\n\n")
df$early <- as.integer(df$period_end >= as.Date("2022-10-01") &
                       df$period_end < as.Date("2024-01-01"))
df$advanced <- as.integer(df$period_end >= as.Date("2024-01-01"))

for (d in list(list(df[df$sme_200 == 1, ], "SME $200M"), list(df[df$sme_500 == 1, ], "SME $500M"))) {
  dat <- d[[1]]; label <- d[[2]]
  dat$tx_e <- dat$early * dat$rho_n
  dat$tx_a <- dat$advanced * dat$rho_n
  m3 <- feols(ln_revenue ~ tx_e + tx_a | ticker + year_quarter, data = dat, cluster = ~ticker)
  be <- coef(m3)["tx_e"]; ba <- coef(m3)["tx_a"]
  # WCB per coefficient (univariate for each)
  pe <- wcb(feols(ln_revenue ~ tx_e | ticker + year_quarter, data = dat, cluster = ~ticker), dat, "tx_e")
  pa <- wcb(feols(ln_revenue ~ tx_a | ticker + year_quarter, data = dat, cluster = ~ticker), dat, "tx_a")
  intens <- ifelse(be != 0, ba / be, NA)
  cat(sprintf("%s (N=%d): Early=%.4f(WCB=%.3f%s) Adv=%.4f(WCB=%.3f%s) Intens=%.2fx\n",
    label, n_distinct(dat$ticker), be, pe, sig(pe), ba, pa, sig(pa), intens))
}

# ============================================
# QUARTILE DiD (Q75 vs Q25)
# ============================================
cat("\n=== QUARTILE DiD ===\n")
pq <- sme200[!duplicated(sme200$ticker), ]
q75 <- quantile(pq$rho, 0.75)
q25 <- quantile(pq$rho, 0.25)
cat(sprintf("Q25=%.1f Q75=%.1f\n", q25, q75))

pq2 <- sme200[sme200$rho >= q75 | sme200$rho <= q25, ]
pq2$high <- as.integer(pq2$rho >= q75)
pq2$tx_bin <- pq2$post * pq2$high
mq <- feols(ln_revenue ~ tx_bin | ticker + year_quarter, data = pq2, cluster = ~ticker)
pwq <- wcb(mq, pq2, "tx_bin")
cat(sprintf("Quartile DiD: beta=%.4f WCB=%.3f%s\n", coef(mq)["tx_bin"], pwq, sig(pwq)))
