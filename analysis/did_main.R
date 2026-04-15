# ================================================
# Main DiD Analysis — Final Version
# "Generative AI as a Task Shock"
# ================================================
# PRIMARY: Literature Rubric (continuous, SME $200M)
# ROBUSTNESS: SME $500M, LLM Judge, O*NET alpha
# ================================================

library(fixest)
library(dplyr)

panel  <- read.csv("data/processed/master_panel.csv")
llm    <- read.csv("data/processed/llm_judge_scores.csv")
lit    <- read.csv("data/processed/lit_continuous_scores.csv")
onet   <- read.csv("data/processed/onet_similarity_scores.csv")

df <- merge(panel,
    llm[,c("ticker","early_ai")], by="ticker")
df$year_quarter <- paste0(
    df$fiscal_year,"Q",df$fiscal_quarter)
df$early_n <- df$early_ai / 100

pre_rev <- df[df$fiscal_year<=2022,] %>%
    group_by(ticker) %>%
    summarise(pre_rev_m=median(revenue/1e6,
        na.rm=TRUE), .groups="drop")
df <- merge(df, pre_rev, by="ticker")
df <- merge(df,
    lit[,c("ticker","lit_score")],
    by="ticker", all.x=TRUE)
df <- merge(df,
    onet[,c("ticker","firm_alpha")],
    by="ticker", all.x=TRUE)
df$lit_n   <- df$lit_score / 100
df$onet_n  <- df$firm_alpha / 0.5

# Samples
sme200 <- df[df$pre_rev_m < 200,]
sme500 <- df[df$pre_rev_m < 500,]

cat("SME $200M:", n_distinct(sme200$ticker), "firms\n")
cat("SME $500M:", n_distinct(sme500$ticker), "firms\n\n")

# WCB (null imposed)
wcb <- function(model, data, xvar, B=999) {
    obs  <- coef(model)[1]
    yvar <- all.vars(formula(model))[1]
    data$y0 <- data[[yvar]] - obs * data[[xvar]]
    m0 <- feols(as.formula(paste("y0 ~", xvar,
        "| ticker + year_quarter")),
        data=data, cluster=~ticker)
    res   <- residuals(m0)
    firms <- unique(data$ticker)
    set.seed(42)
    boot <- replicate(B, {
        w <- sample(c(-1,1),length(firms),replace=T)
        names(w) <- firms
        data$wt  <- w[data$ticker]
        data$yb  <- fitted(m0) + res * data$wt
        coef(feols(as.formula(paste("yb ~", xvar,
            "| ticker + year_quarter")),
            data=data, cluster=~ticker))[1]
    })
    mean(abs(boot) >= abs(obs))
}

sig <- function(p) ifelse(p<0.01,"***",
    ifelse(p<0.05,"**",ifelse(p<0.10,"*","")))

# ---- TABLE 1: PRIMARY ----
cat("=== TABLE 1: PRIMARY RESULTS ===\n\n")

p200 <- sme200[sme200$fiscal_year>=2020 &
    !is.na(sme200$lit_score),]
p200$post_x_lit <- p200$post * p200$lit_n

m1 <- feols(ln_revenue ~ post_x_lit |
    ticker + year_quarter,
    data=p200, cluster=~ticker)
p1 <- wcb(m1, p200, "post_x_lit")
cat(sprintf("Lit Rubric (SME $200M, N=%d):\n",
    n_distinct(p200$ticker)))
cat(sprintf("  beta=%.4f SE=%.4f clust_p=%.3f WCB=%.3f%s\n\n",
    coef(m1)[1], se(m1)[1],
    pvalue(m1)[1], p1, sig(p1)))

p500 <- sme500[sme500$fiscal_year>=2020 &
    !is.na(sme500$lit_score),]
p500$post_x_lit <- p500$post * p500$lit_n

m1b <- feols(ln_revenue ~ post_x_lit |
    ticker + year_quarter,
    data=p500, cluster=~ticker)
p1b <- wcb(m1b, p500, "post_x_lit")
cat(sprintf("Lit Rubric (SME $500M, N=%d):\n",
    n_distinct(p500$ticker)))
cat(sprintf("  beta=%.4f SE=%.4f clust_p=%.3f WCB=%.3f%s\n\n",
    coef(m1b)[1], se(m1b)[1],
    pvalue(m1b)[1], p1b, sig(p1b)))

# ---- TABLE 2: ROBUSTNESS ----
cat("=== TABLE 2: ROBUSTNESS ===\n\n")

# LLM Judge
p_llm <- sme200[sme200$fiscal_year>=2020,]
p_llm$post_x_llm <- p_llm$post * p_llm$early_n
m2 <- feols(ln_revenue ~ post_x_llm |
    ticker + year_quarter,
    data=p_llm, cluster=~ticker)
p2 <- wcb(m2, p_llm, "post_x_llm")
cat(sprintf("LLM Judge (SME $200M, N=%d):\n",
    n_distinct(p_llm$ticker)))
cat(sprintf("  beta=%.4f WCB=%.3f%s\n\n",
    coef(m2)[1], p2, sig(p2)))

# O*NET
p_onet <- sme200[sme200$fiscal_year>=2020 &
    !is.na(sme200$firm_alpha),]
p_onet$post_x_onet <- p_onet$post * p_onet$onet_n
m3 <- feols(ln_revenue ~ post_x_onet |
    ticker + year_quarter,
    data=p_onet, cluster=~ticker)
p3 <- wcb(m3, p_onet, "post_x_onet")
cat(sprintf("O*NET alpha (SME $200M, N=%d):\n",
    n_distinct(p_onet$ticker)))
cat(sprintf("  beta=%.4f WCB=%.3f%s\n\n",
    coef(m3)[1], p3, sig(p3)))

# ---- PLACEBO ----
cat("=== PLACEBO ===\n")
pl <- sme200[sme200$fiscal_year>=2020 &
    sme200$fiscal_year<=2022 &
    !is.na(sme200$lit_score),]
pl$post_pl <- as.integer(
    pl$fiscal_year==2021 |
    (pl$fiscal_year==2022 & pl$fiscal_quarter<=3))
pl$post_x_pl <- pl$post_pl * pl$lit_n
mpl <- feols(ln_revenue~post_x_pl|
    ticker+year_quarter,
    data=pl, cluster=~ticker)
p_pl <- wcb(mpl, pl, "post_x_pl")
cat(sprintf("Lit Rubric placebo WCB: %.3f%s\n\n",
    p_pl, sig(p_pl)))

# ---- SIZE HETEROGENEITY ----
cat("=== SIZE HETEROGENEITY ===\n")
cat(sprintf("%-12s %5s %8s %8s\n",
    "Cutoff","N","Beta","WCB_p"))
cat(strrep("-",36),"\n")
for(cut in c(Inf, 500, 300, 200)){
    p_c <- df[df$fiscal_year>=2020 &
        !is.na(df$lit_score) &
        (if(is.infinite(cut)) TRUE
         else df$pre_rev_m<cut),]
    p_c$tx <- p_c$post * p_c$lit_n
    mc <- feols(ln_revenue~tx|
        ticker+year_quarter,
        data=p_c, cluster=~ticker)
    pc <- wcb(mc, p_c, "tx")
    label <- if(is.infinite(cut)) "Full" else
        paste0("<$",cut,"M")
    cat(sprintf("%-12s %5d %8.4f %8.3f%s\n",
        label, n_distinct(p_c$ticker),
        coef(mc)[1], pc, sig(pc)))
}

# ---- THREE-PERIOD ANALYSIS ----
cat("\n=== THREE-PERIOD ANALYSIS ===\n\n")

df_3p <- df
df_3p$early_ai_d <- as.integer(
    (df_3p$fiscal_year==2022 & df_3p$fiscal_quarter==4) |
    df_3p$fiscal_year==2023)
df_3p$adv_ai_d <- as.integer(df_3p$fiscal_year>=2024)

cat(sprintf("%-12s %5s %8s %8s %8s %8s\n",
    "Sample","N","Beta_E","WCB_E","Beta_A","WCB_A"))
cat(strrep("-",55),"\n")

for(cutoff in c(200,500)){
    p3 <- df_3p[df_3p$fiscal_year>=2020 &
        !is.na(df_3p$lit_score) &
        df_3p$pre_rev_m<cutoff,]
    p3$tx_e <- p3$early_ai_d * p3$lit_n
    p3$tx_a <- p3$adv_ai_d   * p3$lit_n
    
    m3 <- feols(ln_revenue ~ tx_e + tx_a |
        ticker + year_quarter,
        data=p3, cluster=~ticker)
    
    be <- coef(m3)["tx_e"]
    ba <- coef(m3)["tx_a"]
    
    # WCB joint null
    yvar <- "ln_revenue"
    m0_3 <- feols(ln_revenue ~ 1 |
        ticker + year_quarter,
        data=p3, cluster=~ticker)
    r3 <- residuals(m0_3)
    firms3 <- unique(p3$ticker)
    set.seed(42)
    boot3 <- replicate(999,{
        w <- sample(c(-1,1),length(firms3),replace=T)
        names(w) <- firms3
        p3$wt <- w[p3$ticker]
        p3$yb <- fitted(m0_3)+r3*p3$wt
        cb <- coef(feols(yb~tx_e+tx_a|
            ticker+year_quarter,
            data=p3,cluster=~ticker))
        cb[c("tx_e","tx_a")]
    })
    pwe <- mean(abs(boot3["tx_e",])>=abs(be))
    pwa <- mean(abs(boot3["tx_a",])>=abs(ba))
    
    cat(sprintf("SME $%dM      %5d %8.4f %8.3f%s %8.4f %8.3f%s\n",
        cutoff, n_distinct(p3$ticker),
        be, pwe, sig(pwe),
        ba, pwa, sig(pwa)))
    cat(sprintf("             Intensification: %.2fx\n",
        ba/be))
}

# ---- QUARTILE ROBUSTNESS (Lit Rubric) ----
cat("\n=== QUARTILE ROBUSTNESS (Lit Rubric) ===\n\n")

p_q <- df[df$fiscal_year>=2020 &
    !is.na(df$lit_score) &
    df$pre_rev_m<200,]

q75 <- quantile(p_q$lit_score[!duplicated(p_q$ticker)], 0.75)
q25 <- quantile(p_q$lit_score[!duplicated(p_q$ticker)], 0.25)

cat(sprintf("Q25=%.0f, Q75=%.0f\n", q25, q75))

p_q2 <- p_q[(p_q$lit_score>=q75 | p_q$lit_score<=q25),]
p_q2$high <- as.integer(p_q2$lit_score>=q75)
p_q2$post_x_bin <- p_q2$post * p_q2$high

mq <- feols(ln_revenue~post_x_bin|
    ticker+year_quarter,
    data=p_q2, cluster=~ticker)
pwq <- wcb(mq, p_q2, "post_x_bin")

n_h <- n_distinct(p_q2$ticker[p_q2$high==1])
n_l <- n_distinct(p_q2$ticker[p_q2$high==0])

cat(sprintf("Quartile DiD: beta=%.4f SE=%.4f WCB=%.3f%s (N_H=%d N_L=%d)\n",
    coef(mq)[1], se(mq)[1], pwq, sig(pwq), n_h, n_l))
