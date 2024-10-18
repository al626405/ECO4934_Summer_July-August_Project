import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Define file paths for the CSV files
train_file = 'TeamB/Data/Exclude_With_Lasso/Train.dat'
test_file = 'TeamB/Data/Exclude_With_Lasso/Test.dat'
validation_file = 'TeamB/Data/Exclude_With_Lasso/Val.dat'
roc_csv_file = 'TeamB/Data/ROC_Data/Boosting_ROCData.csv'

# Load data from CSV files
Train = pd.read_csv(train_file, header=None)
Test = pd.read_csv(test_file, header=None)
Validation = pd.read_csv(validation_file, header=None)

# Check the number of columns
print(f"Number of columns in Train: {Train.shape[1]}")
print(f"Number of columns in Test: {Test.shape[1]}")
print(f"Number of columns in Validation: {Validation.shape[1]}")

# Define the number of features and create column names
num_features = Train.shape[1] - 1  # Number of features
feature_columns = [f'feature_{i}' for i in range(num_features)]
target_column = 1  # Assuming the second column is the target

# Prepare data for training
X_train = Train.drop(columns=[target_column])
y_train = Train[target_column]

# Ensure the target variable is integer type
y_train = y_train.astype(int)

# Define the Gradient Boosting model with custom parameters
gb_model = GradientBoostingClassifier(
    n_estimators=1500,       # Number of boosting stages
    learning_rate=0.1,       # Step size
    max_depth=7,             # Maximum depth of the individual trees
    subsample=0.8,           # Fraction of samples used to fit each tree
    random_state=123         # Random seed for reproducibility
)

# Train the model using cross-validation
cv_scores = cross_val_score(gb_model, X_train, y_train, cv=5)

# Fit the model on the entire training set
gb_model.fit(X_train, y_train)

# Print the model's cross-validation scores
print("Cross-validation scores: ", cv_scores)
print("Average cross-validation score: ", cv_scores.mean())

# Predict using the gradient boosting model on the validation set
X_validation = Validation.drop(columns=[target_column])
y_validation = Validation[target_column]

# Ensure the target variable is integer type
y_validation = y_validation.astype(int)

gb_pred = gb_model.predict_proba(X_validation)[:, 1]

# Evaluate the model using ROC curve
fpr, tpr, _ = roc_curve(y_validation, gb_pred)
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
plt.title('Gradient Boosting ROC Curve')
plt.legend(loc="lower right")
plt.savefig("TeamB/Figures/Boosting_ROC.jpg")

# Save ROC curve data to CSV, including only FPR, TPR, and AUC
roc_data = pd.DataFrame({
    'FPR': fpr, 
    'TPR': tpr,
    'Model': "Boosting",
})
roc_data['AUC'] = roc_auc  # Add AUC to the DataFrame
roc_data.to_csv(roc_csv_file, index=False)
print(f"ROC curve data saved to {roc_csv_file}")
