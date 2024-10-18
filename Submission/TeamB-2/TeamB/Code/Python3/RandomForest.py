#Alexis Leclerc
#07/18/2024

import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


# Define file paths for the CSV files
train_file = 'TeamB/Data/Exclude_With_Lasso/Train.dat'
test_file = 'TeamB/Data/Exclude_With_Lasso/Test.dat'
validation_file = 'TeamB/Data/Exclude_With_Lasso/Val.dat'
roc_csv_file = 'TeamB/Data/ROC_Data/RandomForest_ROCData.csv'

# Load data from CSV files
Train = pd.read_csv(train_file, header=None)
Test = pd.read_csv(test_file, header=None)
Validation = pd.read_csv(validation_file, header=None)

num_features = Train.shape[1] - 1  # Number of features
feature_columns = [f'feature_{i}' for i in range(num_features)]
target_column = 1  # Assuming the second column is the target

# Prepare data for training
X_train = Train.drop(columns=[target_column])
y_train = Train[target_column]

# Ensure the target variable is integer type
y_train = y_train.astype(int)



# Train the random forest model using cross-validation
rf_model = RandomForestClassifier(random_state=123)
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)

# Fit the model on the entire training set
rf_model.fit(X_train, y_train)

# Print the model's cross-validation scores
print("Cross-validation scores: ", cv_scores)
print("Average cross-validation score: ", cv_scores.mean())

# Predict using the random forest model on the test set
X_Test = Test.drop(columns=[target_column])
y_Test= Test[target_column]

rf_pred = rf_model.predict_proba(X_Test)[:, 1]

# Evaluate the model using ROC curve
fpr, tpr, thresholds = roc_curve(y_Test, rf_pred)
roc_auc = auc(fpr, tpr)

# Print and plot the ROC curve
print(f"ROC AUC: {roc_auc}")

plt.figure()
plt.plot(fpr, tpr, color='purple', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC Curve')
plt.legend(loc="lower right")
plt.savefig("TeamB/Figures/RandomForest_ROC.jpg")

#Write to CSV

RandomForestROC = {
    'FPR': fpr,
    'TPR': tpr,
    'Model': "Random Forest",
    'AUC': [roc_auc] * len(fpr)  # AUC is the same for all points
}

# Create a DataFrame
RfROC = pd.DataFrame(RandomForestROC)

# Save to CSV
RfROC.to_csv(roc_csv_file, index=False)



