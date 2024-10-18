# Alexis Leclerc
# 07/18/2024

rm(list = ls())
gc()

# Load required libraries
library(caret)
library(parallel)
library(doParallel)
library(foreach)
library(pROC)

# Set up parallel processing
num_cores <- detectCores() - 1
registerDoParallel(num_cores)

# Read data from CSV files without headers
Train <- read.table("TeamB/Data/Exclude_With_Lasso/Train.dat", header = FALSE, sep = ",")
Test <- read.table("TeamB/Data/Exclude_With_Lasso/Test.dat", header = FALSE, sep = ",")
Validation <- read.table("TeamB/Data/Exclude_With_Lasso/Val.dat", header = FALSE, sep = ",")

# Rename columns for clarity
colnames(Train) <- paste0("V", 1:ncol(Train))
colnames(Test) <- paste0("V", 1:ncol(Test))
colnames(Validation) <- paste0("V", 1:ncol(Validation))

# Ensure the response variable is a factor
Train$V2 <- as.factor(Train$V2)

# Train a Bagging model
bagging_model <- train(V2 ~ ., data = Train, method = "treebag",
                       trControl = trainControl(method = "cv", number = 5, allowParallel = TRUE))

# Prepare data for prediction
YTest <- Test$V2
YVal <- Validation$V2

XTest <- Test[, !names(Test) %in% "V2"]
XVal <- Validation[, !names(Validation) %in% "V2"]

# Predict using the Bagging model
bagging_pred <- predict(bagging_model, newdata = XTest, type = "prob")[, 2]

# Calculate ROC for Bagging model
bagging_ROC <- roc(YTest, bagging_pred)

# Plot and save the ROC curve to PDF
jpeg("TeamB/Figures/Bagging_ROC.jpg")
plot.roc(bagging_ROC, legacy.axes = TRUE, print.auc = TRUE, print.auc.y = 0.52, print.auc.x = 0.75,
         xlim = c(1, 0), asp = NA, xlab = "(1-Specificity), FPR", ylab = "Sensitivity, TPR",
         main = "Bagging ROC Curve", col = "purple")

# Additional annotations
text(0.4, 0.4, "AUC = 0.50", col = "red", cex = 1, font = 2)
abline(a = 1, b = -1, col = "red", lty = 1)

# Close the PDF device
dev.off()

BaggingROC <- data.frame(
  FPR = c(1 - bagging_ROC$specificities),
  TPR = c(bagging_ROC$sensitivities),
  Model = rep(c("Bagging"), each = length(bagging_ROC$sensitivities)),
  AUC = c(bagging_ROC$auc)
)

write.csv(BaggingROC, "TeamB/Data/ROC_Data/Bagging_ROCData.csv", row.names = FALSE)

# Stop parallel processing
stopImplicitCluster()


rm(list = ls())
gc()
quit(save = "no")
