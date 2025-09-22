🌟 Loan Approval Classification
This project applies and compares two distinct classification algorithms—K-Nearest Neighbors (KNN) and Support Vector Machine (SVM)—to predict loan approval status based on various applicant features. The goal is to identify which model provides the best performance for this specific dataset.

The core of this project is a data science pipeline that starts with exploratory data analysis (EDA), followed by data preprocessing, and concludes with the application of machine learning models. Using a dataset of 45,000 loan applications, the program analyzes key features such as a person's income, age, credit history length, and interest rates to build a predictive model.

The program addresses a common challenge in real-world datasets: class imbalance. The loan_status variable, which is the target for prediction, is not evenly distributed, with the minority class representing only 22.22% of the dataset. To handle this, the project employs the SMOTE (Synthetic Minority Over-sampling Technique) algorithm to balance the training data, ensuring the models can learn effectively from both approved and rejected loan applications.

The final models are evaluated using classification reports and confusion matrices to compare their accuracy, precision, and recall, providing a clear and comprehensive assessment of their performance.

✨ Key Features & Technical Details
Exploratory Data Analysis (EDA): The notebook includes an in-depth analysis of the dataset, with a focus on understanding data types, distributions, and the presence of any missing or duplicate values. The correlation between numerical features is visualized using a heatmap .

Data Preprocessing: It cleans the data and handles outliers in the person_age feature using the Interquartile Range (IQR) method. The program also converts categorical features like person_gender into a numerical format (female to 0, male to 1) for the models.

Class Imbalance Mitigation: The SMOTE algorithm is used to oversample the minority class, ensuring the models are trained on a balanced dataset.

Feature Engineering & Scaling: A select subset of key features (loan_int_rate, loan_percent_income, person_age, and cb_person_cred_hist_length) is used for classification, and these features are scaled using StandardScaler to ensure all numerical features have a consistent range.

Model Implementation & Hyperparameter Tuning:

K-Nearest Neighbors (KNN): The model is fine-tuned using GridSearchCV to identify the optimal parameters (n_neighbors, weights, and metric) for the highest accuracy. The best-performing model achieved an accuracy of approximately 80.92% during cross-validation.

Support Vector Machine (SVM): A linear kernel SVM model is also implemented for a direct comparison with KNN. This model achieved an accuracy of approximately 74.01%.

Model Evaluation: Both models are evaluated with a classification report and a visual confusion matrix, which clearly show the strengths and weaknesses of each algorithm on the specific dataset .

🚀 Getting Started
To run this project, you will need a Python environment with the following libraries:

pandas

numpy

matplotlib

seaborn

scikit-learn

imblearn (imbalanced-learn)

You can set up the environment and run the analysis by cloning the repository and executing the Classificationproject.ipynb file in a Jupyter Notebook environment.

📊 Project Workflow
The Classificationproject.ipynb notebook follows a structured machine learning workflow:

Data Loading & Inspection: The program begins by importing necessary libraries and loading the Loan_Approval.csv dataset. It then conducts an initial inspection of the data, using df.info() to check data types and df.head() to understand the data structure. df.describe() is used to provide a statistical summary of the numerical features.

Data Cleaning & Exploration: The project confirms there are no missing values in the dataset using df.isnull().sum(). It also checks for duplicate rows, but none are found. The script then transforms categorical features like person_gender and ensures all numerical features are of the appropriate integer type.

Visualization & Feature Selection: The script generates visualizations to better understand the data. A histogram of person_age is created to show its distribution, and a pair plot is used to explore relationships between key features like loan_int_rate, loan_percent_income, and credit_score. A correlation heatmap is also generated to visualize the relationships between all numerical features. The features for the model are selected, including loan_int_rate, loan_percent_income, person_age, and cb_person_cred_hist_length.

Data Preparation: The data is split into training and testing sets. Outliers in the person_age feature are removed using the IQR method. Finally, the SMOTE algorithm is applied to the training data to address the class imbalance issue. All features are then scaled using StandardScaler.

Model Building & Evaluation:

KNN: A KNeighborsClassifier is initialized and tuned with GridSearchCV to find the best parameters. The model is then trained on the scaled and oversampled data.

SVM: A SVC model with a linear kernel is trained on the same prepared dataset.

Comparison: Predictions are made on the test data for both models. The classification_report and confusion_matrix are used to evaluate their performance, providing a comprehensive comparison of the two algorithms' effectiveness in predicting loan approval.

📈 Final Thoughts
The analysis demonstrates the effectiveness of machine learning for loan approval classification. The KNN model, after careful hyperparameter tuning, proved to be a more accurate predictor than the linear SVM model on this dataset. This project serves as a solid foundation for further exploration, such as experimenting with additional features or different models like Random Forest or Gradient Boosting.

🙏 Acknowledgments
I extend my thanks to the creators of the pandas, scikit-learn, imblearn, matplotlib, and seaborn libraries for providing the powerful tools that made this analysis possible.
