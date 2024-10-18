# Alexis Leclerc
# 07/17/2024

#Getting them packages installed
install.packages("foreach", repos = "https://cran.rstudio.com/")
install.packages("doParallel", repos = "https://cran.rstudio.com/")
install.packages("remotes")
remotes::install_version("glmnet", version = "3.0-0")
install.packages("pROC", repos = "https://cran.rstudio.com/")
install.packages("ROCR", repos = "https://cran.rstudio.com/")
install.packages("caret", repos = "https://cran.rstudio.com/")
install.packages("parallel", repos = "https://cran.rstudio.com/")
install.packages("rpart", repos = "https://cran.rstudio.com/")
install.packages("rpart.plot", repos = "https://cran.rstudio.com/")


rm(list = ls())
gc()

# Load required libraries
library(foreach)
library(doParallel)
library(glmnet)
library(pROC)
library(ROCR)

# Set up parallel processing
num_cores <- parallel::detectCores() - 1 
registerDoParallel(num_cores)

# Read data from CSV files without headers
TrainLasso <- read.table("TeamB/Data/Exclude_With_Lasso/Train.dat", header = FALSE, sep = ",")
TestLasso <- read.table("TeamB/Data/Exclude_With_Lasso/Test.dat", header = FALSE, sep = ",")
ValidationLasso <- read.table("TeamB/Data/Exclude_With_Lasso/Val.dat", header = FALSE, sep = ",")

TrainLogit <- read.table("TeamB/Data/Exclude_Without_Lasso/Train.dat", header = FALSE, sep = ",")
TestLogit <- read.table("TeamB/Data/Exclude_Without_Lasso/Test.dat", header = FALSE, sep = ",")
ValidationLogit <- read.table("TeamB/Data/Exclude_Without_Lasso/Val.dat", header = FALSE, sep = ",")

# Update column names after re-importing data
colnames(TrainLasso) <- paste0("V", 1:ncol(TrainLasso))
colnames(TestLasso) <- paste0("V", 1:ncol(TestLasso))
colnames(ValidationLasso) <- paste0("V", 1:ncol(ValidationLasso))

colnames(TrainLogit) <- paste0("V", 1:ncol(TrainLogit))
colnames(TestLogit) <- paste0("V", 1:ncol(TestLogit))
colnames(ValidationLogit) <- paste0("V", 1:ncol(ValidationLogit))

# Separate features and response variable for Exclude_Without_Lasso
XTrainLogit <- as.matrix(TrainLogit[, -2])
XTestLogit <- as.matrix(TestLogit[, -2])
XValLogit <- as.matrix(ValidationLogit[, -2])

YTrainLogit <- TrainLogit[, 2]  # Response variable
YTestLogit <- TestLogit[, 2]
YValLogit <- ValidationLogit[, 2]


# Logit model on Exclude_Without_Lasso data
logit_data <- as.data.frame(cbind(YTrainLogit, XTrainLogit))
Logit_model <- glm(YTrainLogit ~ ., data = logit_data, family = binomial(link = "logit"))

# Predict using the Logit model
logit_pred <- predict(Logit_model, newdata = as.data.frame(XTestLogit), type = "response")

# Calculate ROC
Logit_ROC <- roc(YTestLogit, logit_pred)

# Plot and save ROC curve to PDF
jpeg("TeamB/Figures/Logit_ROC.jpg")
plot.roc(Logit_ROC, legacy.axes = TRUE, print.auc = TRUE, print.auc.y = 0.58, print.auc.x = 0.75,
         xlim = c(1, 0), asp = NA, xlab = "(1-Specificity), FPR", ylab = "Sensitivity, TPR",
         main = "Logit ROC Curve", col = "blue")
dev.off()

# Update column names for Exclude_With_Lasso
colnames(TrainLasso) <- paste0("V", 1:ncol(TrainLasso))
colnames(TestLasso) <- paste0("V", 1:ncol(TestLasso))
colnames(ValidationLasso) <- paste0("V", 1:ncol(ValidationLasso))

# Separate features and response variable for Exclude_With_Lasso
XTrainLasso <- as.matrix(TrainLasso[, -2])
XTestLasso <- as.matrix(TestLasso[, -2])
XValLasso <- as.matrix(ValidationLasso[, -2])

YTrainLasso <- TrainLasso[, 2]  # Response variable
YTestLasso <- TestLasso[, 2]
YValLasso <- ValidationLasso[, 2]


# Logit-Lasso model on Exclude_With_Lasso data
Lasso_data <- as.data.frame(cbind(YTrainLasso, XTrainLasso))
Lasso_model <- glm(YTrainLasso ~ ., data = Lasso_data, family = binomial(link = "logit"))

# Predict using the Logit-Lasso model
Lasso_pred <- predict(Lasso_model, newdata = as.data.frame(XTestLasso), type = "response")

# Calculate ROC
Lasso_ROC <- roc(YTestLasso, Lasso_pred)

# Plot and save ROC curve to PDF
jpeg("TeamB/Figures/Logit-Lasso_ROC.jpg")
plot.roc(Lasso_ROC, legacy.axes = TRUE, print.auc = TRUE, print.auc.y = 0.58, print.auc.x = 0.75,
         xlim = c(1, 0), asp = NA, xlab = "(1-Specificity), FPR", ylab = "Sensitivity, TPR",
         main = "Logit-Lasso ROC Curve", col = "blue")
dev.off()

LogitLassoROC <- data.frame(
  FPR = c((1 - Logit_ROC$specificities), (1 - Lasso_ROC$specificities)),
  TPR = c(Logit_ROC$sensitivities, Lasso_ROC$sensitivities),
  Model = rep(c("Logit", "Logit with LASSO"), each = length(Logit_ROC$sensitivities)),
  AUC = c(Logit_ROC$auc, Lasso_ROC$auc)
)

write.csv(LogitLassoROC, "TeamB/Data/ROC_Data/Logit-Lasso_ROCData.csv", row.names = FALSE)

# Stop parallel processing
stopImplicitCluster()

rm(list = ls())
gc()
quit(save = "no")

