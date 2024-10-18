#Alexis Leclerc
#07/31/2024

rm(list = ls())
gc()

library(caret)
library(parallel)
library(doParallel)
library(foreach)
library(rpart)
library(rpart.plot)
library(pROC)

num_cores <- detectCores() - 1 
registerDoParallel(num_cores)

# Read data with header = FALSE since there are no headers
Train <- read.csv("TeamB/Data/Exclude_With_Lasso/Train.dat", header = FALSE)
Test <- read.csv("TeamB/Data/Exclude_With_Lasso/Test.dat", header = FALSE)
Validation <- read.csv("TeamB/Data/Exclude_With_Lasso/Val.dat", header = FALSE)

# Assign proper column names for clarity
colnames(Train) <- paste0("V", 1:ncol(Train))
colnames(Test) <- paste0("V", 1:ncol(Test))
colnames(Validation) <- paste0("V", 1:ncol(Validation))

# Extract the response variable and feature matrix
YTrain <- as.factor(Train$V2)
XTrain <- Train[, -2]
YTest <- as.factor(Test$V2)
XTest <- Test[, -2]

# Create the decision tree model
tree_model <- rpart(V2 ~ ., data = Train, method = "class", cp = 0)

# Use parallel processing to make predictions
tree_pred <- foreach(i = 1:nrow(Test), .combine = c, .packages = 'rpart') %dopar% {
  predict(tree_model, newdata = Test[i, , drop = FALSE], type = "prob")[, 2]
}

stopImplicitCluster()

# Calculate ROC curve
tree_ROC <- roc(YTest, tree_pred)

# Save the ROC plot to a PDF
jpeg("TeamB/Figures/DecisionTree_ROC.jpg")

plot.roc(tree_ROC, legacy.axes = TRUE, print.auc = TRUE, print.auc.y = 0.58, print.auc.x = 0.75,
         xlim = c(1, 0), asp = NA, xlab = "(1-Specificity), FPR", ylab = "Sensitivity, TPR",
         main = "Decision Tree ROC Curve", col = "darkgreen")

legend("bottomright", legend = c("Decision Tree", "Random Guessing"),
       col = c("darkgreen", "red"), lty = 1, cex = 0.8)
text(0.4, 0.4, paste("AUC = 0.50", col = "red", cex = 1, font = 2))
abline(a = 1, b = -1, col = "red", lty = 1)

dev.off()

TreeROC <- data.frame(
  FPR = c(1 - tree_ROC$specificities),
  TPR = c(tree_ROC$sensitivities),
  Model = rep(c("Decision Tree"), each = length(tree_ROC$sensitivities)),
  AUC = c(tree_ROC$auc)
)


write.csv(TreeROC, "TeamB/Data/ROC_Data/DecisionTree_ROCData.csv", row.names = FALSE)

stopImplicitCluster()

rm(list = ls())
gc()
quit(save = "no")
