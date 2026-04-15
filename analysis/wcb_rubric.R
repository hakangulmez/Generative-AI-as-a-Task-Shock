library(fixest)
library(dplyr)

panel  <- read.csv("data/processed/master_panel.csv")
scores <- read.csv("data/processed/rubric_v2_scores.csv")

merged <- merge(panel,
    scores[,c("ticker","rubric_v2")],
    by="ticker")
merged$post_x_rv2 <- merged$post * merged$rubric_v2
merged$year_quarter <- paste0(
    merged$fiscal_year,"Q",merged$fiscal_quarter)
merged$revenue_m <- merged$revenue * 1e-6

wb_firms <- unique(merged$ticker[
    merged$text_source=="wayback"])
panel_20 <- merged[
    merged$ticker %in% wb_firms &
    merged$fiscal_year >= 2020,]

set.seed(42)
B <- 9999

cat("=== WILD CLUSTER BOOTSTRAP B=9999 ===\n\n")

# Substitution
m1 <- feols(ln_revenue ~ post_x_rv2 |
    ticker + year_quarter,
    data=panel_20, cluster=~ticker)
boot1 <- wildboottest(m1,
    param="post_x_rv2", B=B,
    type="rademacher")
cat("Substitution ln(Rev):\n")
cat("  beta:", round(coef(m1)[1],4),"\n")
cat("  WCB p:", round(boot1$p_val,4),"\n\n")

# Commodification
m3 <- feols(gross_margin ~ post_x_rv2 |
    ticker + year_quarter,
    data=panel_20, cluster=~ticker)
boot3 <- wildboottest(m3,
    param="post_x_rv2", B=B,
    type="rademacher")
cat("Commodification GM:\n")
cat("  beta:", round(coef(m3)[1],4),"\n")
cat("  WCB p:", round(boot3$p_val,4),"\n")
