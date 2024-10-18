# ECO4934 
## SUMMER 2024 Project (July - August 2024)

#### TEAM MEMBERS:
-  **Polina Baikova**
-  **Payton Irvin**
-  **Jonathon Lewis**
-  **Alexis Leclerc**

Together we explored the Eclipse and Mozilla Defect Tracking Dataset, which contains data on reported bugs for these four products: Eclipse Platform, the Java Development Tools (JDT), C/C++ Development Tools (CDT) and Plug-­‐in Development Environment (PDE). Tasks we completed include cleaning and filtering the data, split data into Training, Testing and Validation datasets, created and stored the data in a MySQL database, created variables by transforming our data allowing us to increase our models predictive power. Additionally, we wrote code in Python and R to run machine learning classification models including Logit-Lasso, Decision Trees, Random Forrest, Bagging, and Gradient Boosting. We generated ROC plots for each model to compare the performance of the models. To display our results we created slides in LaTeX using beamer and webpages in HTML. We automated this process using a Bash make file to clean the data, run the models, export the figures and create the slides and webpages. This project was completed on a Linux Centos 7 server.

We concluded that the Gradient Boosting and Random Forrest models performed the best with around a 0.92 Area Under the Curve (AUC). This shows the predictive power that our models hold on new unseen data. During this project we learned about the struggles of big data and the importance of computational efficiency. Additionally we grasped an understanding of multiple programming languages and their purpose in the project. As a team we also gained insights on the importance of automation and incorporating it in our scripts through Bash, R and Python.
