import pandas as pd
import matplotlib.pyplot as plt

# File paths for the CSV files containing the ROC data
roc_csv_files = [
    'TeamB/Data/ROC_Data/Logit-Lasso_ROCData.csv',
    'TeamB/Data/ROC_Data/DecisionTree_ROCData.csv',
    'TeamB/Data/ROC_Data/Bagging_ROCData.csv',
    'TeamB/Data/ROC_Data/RandomForest_ROCData.csv',
    'TeamB/Data/ROC_Data/Boosting_ROCData.csv',
]

# Labels for the ROC curves
roc_labels = ['Logit-Lasso', 'Decision Tree', 'Bagging', 'Random Forest', 'Boosting']

# Loop over each CSV file and plot the ROC curve
for file, label in zip(roc_csv_files, roc_labels):
    # Read the CSV file
    roc_data = pd.read_csv(file)

    # Assuming the CSV has columns named 'FPR' and 'TPR'
    fpr = roc_data['FPR']
    tpr = roc_data['TPR']

    # Plot the ROC curve
    plt.plot(fpr, tpr, label=label)

# Plot diagonal line for random guessing
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')

# Set plot labels and title
plt.xlabel('(1 - Specificity), FPR')
plt.ylabel('Sensitivity, TPR')
plt.title('ROC Curves')
plt.legend()
plt.grid(True)

# Save the plot as a PDF file
plt.savefig('TeamB/Figures/Combined.pdf')
