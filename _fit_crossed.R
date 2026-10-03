# Crossed random effects: language and country are NOT nested (Spanish spans 7
# countries; Spain contains 3 languages), so neither can be a level of the other.
suppressMessages({library(lme4); library(pROC)})
d <- read.csv("_crossed_model_frame.csv", encoding = "UTF-8")
feats <- readLines("_feats.txt")
d$lang <- factor(d$lang); d$country <- factor(d$country); d$source <- factor(d$source)
fml <- as.formula(paste("male ~", paste(feats, collapse = " + "),
                        "+ (1|lang) + (1|country)"))
cat("fitting", length(feats), "fixed effects, crossed (1|lang) + (1|country)\n")
m <- glmer(fml, data = d, family = binomial,
           control = glmerControl(optimizer = "bobyqa",
                                  optCtrl = list(maxfun = 2e5)))
cat("\n--- variance components ---\n")
print(VarCorr(m))
cat("\n--- top fixed effects by |z| ---\n")
s <- summary(m)$coefficients
s <- s[rownames(s) != "(Intercept)", , drop = FALSE]
s <- s[order(-abs(s[, "z value"])), ]
print(round(head(s, 9), 3))
p <- predict(m, type = "response")
cat("\nAUC (unweighted):", round(as.numeric(pROC::auc(d$male, p, quiet = TRUE)), 4), "\n")
cat("accuracy        :", round(mean((p >= .5) == (d$male == 1)), 4), "\n")
