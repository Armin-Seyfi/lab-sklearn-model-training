# sklearn Model Training + Evaluation

> **How you'll submit this lab**
>
> This repo is your lab. Fork it, do the work described below in your fork, then open a pull
> request back into this repository. An AI reviewer will check your PR against `rubric.md` and
> leave feedback directly on the PR. See `README.md` for the full workflow.

**Scenario**  
You're working as a machine learning intern at consulting company. Your team is developing predictive models to help with early disease detection. In Part 1, you'll work with a guided exercise using the breast cancer dataset to learn the fundamentals of machine learning with sklearn. In Part 2, you'll apply these skills independently to predict customer churn for a telecommunications company.

**Learning Objectives**
- [ ] Load and explore datasets using sklearn
- [ ] Split data into training and testing sets
- [ ] Train a K-Nearest Neighbors (KNN) classifier
- [ ] Evaluate model performance using accuracy, precision, recall, and confusion matrix
- [ ] Apply machine learning workflow to a new problem (customer churn)
- [ ] Interpret model results and make recommendations

**Estimated Time:** 150-180 minutes (Part 1: 60-75 min, Part 2: 90-105 min)

**Prerequisites:**
- [ ] Basic understanding of Python and pandas
- [ ] Familiarity with data manipulation (from previous lab)
- [ ] Understanding of basic machine learning concepts
- [ ] Knowledge of train/test split concept

---

## Introduction

Welcome to the sklearn Model Training and Evaluation lab! This lab is divided into two parts:

**Part 1 (Guided):** You'll work through a complete machine learning workflow step-by-step with detailed code and explanations. We'll use the breast cancer dataset to predict whether a tumor is malignant (cancerous) or benign (non-cancerous).

**Part 2 (Unguided):** You'll apply what you learned to predict customer churn for a telecommunications company. You'll receive the dataset and a list of steps to follow, but you'll write the code yourself.

**What you'll build:**
- A KNN classifier for breast cancer prediction
- A customer churn prediction model
- Understanding of the complete ML workflow: load → explore → split → train → evaluate

**Why this matters:**
Machine learning is at the heart of AI applications. Understanding how to train and evaluate models is essential for any AI practitioner. The sklearn library is one of the most widely used tools in the industry, making it crucial to master.

**Success criteria:**
- [ ] Successfully train a KNN model on the breast cancer dataset
- [ ] Achieve reasonable accuracy (>90%) on the test set
- [ ] Understand and interpret evaluation metrics
- [ ] Independently build a churn prediction model
- [ ] Compare model performance and draw conclusions

---

## Part 1: Guided Exercise - Breast Cancer Prediction

### Background

The breast cancer dataset contains measurements of cell nuclei from breast mass samples. Each sample has 30 features (like radius, texture, perimeter, etc.) and a target label indicating whether the tumor is malignant (1) or benign (0).

Your task is to build a model that can predict whether a new tumor sample is malignant or benign based on these measurements. You should start a new notebook and copy the code templates provided in each step. You are encouraged to run the code and experiment with it and are allowed to change the parameters to see how it affects the results. After each step, write your conclusions and observations.

---

### Step 1: Setting Up and Loading Data

**Objective:** Import necessary libraries and load the breast cancer dataset.

**What to do:**
1. Import sklearn, pandas, numpy, and matplotlib
2. Load the breast cancer dataset
3. Explore the dataset structure

**Code template:**
```python
"""
Breast Cancer Prediction with KNN
Author: [Your Name]
Description: Predict whether a breast tumor is malignant or benign using KNN
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

# Load the breast cancer dataset
print("Loading breast cancer dataset...")
cancer_data = load_breast_cancer()

# The dataset is a Bunch object with 'data', 'target', 'feature_names', etc.
print(f"\nDataset type: {type(cancer_data)}")
print(f"Number of samples: {len(cancer_data.data)}")
print(f"Number of features: {len(cancer_data.feature_names)}")
print(f"Target classes: {cancer_data.target_names}")

# Convert to DataFrame for easier manipulation
df = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
df['target'] = cancer_data.target

print("\nFirst few rows:")
print(df.head())

print("\nDataset info:")
print(df.info())

print("\nTarget distribution:")
print(df['target'].value_counts())
print(f"Malignant (1): {(df['target'] == 1).sum()}")
print(f"Benign (0): {(df['target'] == 0).sum()}")
```

Tip: if you are not using the NumPy library, this code will need to be adapted.

**Expected outcome:** You should see 569 samples, 30 features, and a roughly balanced dataset (around 357 benign, 212 malignant).

**Checkpoint:** Verify that your dataset loads correctly and you can see the feature names and target distribution.

---

### Step 2: Data Exploration

**Objective:** Understand the data better through basic statistics and visualization.

**What to do:**
1. Calculate basic statistics for the features
2. Check for missing values
3. Visualize the distribution of a few key features

**Code template:**
```python
# Basic statistics
print("\n" + "="*50)
print("BASIC STATISTICS")
print("="*50)
print(df.describe())

# Check for missing values
print("\n" + "="*50)
print("MISSING VALUES")
print("="*50)
missing = df.isnull().sum()
if missing.sum() == 0:
    print("✓ No missing values found!")
else:
    print(missing[missing > 0])

# Visualize feature distributions (select a few key features)
print("\n" + "="*50)
print("FEATURE DISTRIBUTIONS")
print("="*50)

# Select a few representative features to visualize
key_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area']

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, feature in enumerate(key_features):
    axes[idx].hist(df[df['target'] == 0][feature], alpha=0.5, label='Benign', bins=30)
    axes[idx].hist(df[df['target'] == 1][feature], alpha=0.5, label='Malignant', bins=30)
    axes[idx].set_xlabel(feature)
    axes[idx].set_ylabel('Frequency')
    axes[idx].set_title(f'Distribution of {feature}')
    axes[idx].legend()

plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=150, bbox_inches='tight')
print("Saved visualization to 'feature_distributions.png'")
plt.show()
```

**Expected outcome:** You should see that malignant tumors tend to have larger values for features like radius, perimeter, and area compared to benign tumors.

**Checkpoint:** Verify that you can see the statistical summary and the visualization shows clear differences between benign and malignant samples.

---

### Step 3: Splitting the Data

**Objective:** Split the data into training and testing sets.

**What to do:**
1. Separate features (X) from target (y)
2. Split into training (80%) and testing (20%) sets
3. Verify the split

**Code template:**
```python
# Separate features and target
X = df.drop('target', axis=1)  # All columns except 'target'
y = df['target']  # Target column

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split into training and testing sets
# random_state ensures reproducibility
# stratify=y ensures both sets have similar class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print("\n" + "="*50)
print("DATA SPLIT")
print("="*50)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
print(f"Training features: {X_train.shape[1]}")
print(f"Test features: {X_test.shape[1]}")

# Verify class distribution in both sets
print("\nTraining set target distribution:")
print(y_train.value_counts())
print(f"  Benign (0): {(y_train == 0).sum()} ({(y_train == 0).mean()*100:.1f}%)")
print(f"  Malignant (1): {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.1f}%)")

print("\nTest set target distribution:")
print(y_test.value_counts())
print(f"  Benign (0): {(y_test == 0).sum()} ({(y_test == 0).mean()*100:.1f}%)")
print(f"  Malignant (1): {(y_test == 1).sum()} ({(y_test == 1).mean()*100:.1f}%)")
```

**Expected outcome:** You should have approximately 455 training samples and 114 test samples, with similar class distributions in both sets.

**Checkpoint:** Verify that your train/test split maintains similar proportions of benign and malignant cases in both sets.

---

### Step 4: Training the KNN Model

**Objective:** Train a K-Nearest Neighbors classifier.

**What to do:**
1. Create a KNN classifier instance
2. Train it on the training data
3. Understand what KNN does

**Code template:**
```python
# Create KNN classifier
# n_neighbors=5 means the model will look at the 5 nearest neighbors to make a prediction

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

print("KNN classifier trained successfully!")
print(f"Number of neighbors (k): {knn.n_neighbors}")

```

**Explanation:**
KNN (K-Nearest Neighbors) is a simple but effective classification algorithm. When making a prediction for a new sample, it:
1. Finds the K nearest training samples (based on feature similarity)
2. Looks at the labels of those K neighbors
3. Predicts the most common label among those neighbors

**Expected outcome:** Your model should train without errors.

**Checkpoint:** Verify that the model trains successfully and you understand how KNN works conceptually.

---

### Step 5: Making Predictions

**Objective:** Use the trained model to make predictions on the test set.

**What to do:**
1. Make predictions on the test set
2. Compare predictions with actual values
3. Look at a few examples

**Code template:**
```python
y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)

print(f"\nTraining predictions: {len(y_train_pred)}")
print(f"Test predictions: {len(y_test_pred)}")

```

**Expected outcome:** You should see predictions for each test sample, and most should be correct.

**Checkpoint:** Verify that predictions are being made and you can see how they compare to actual values.

---

### Step 6: Evaluating Model Performance

**Objective:** Calculate and interpret various evaluation metrics.

**What to do:**
1. Calculate accuracy, precision, and recall
2. Create a confusion matrix
3. Generate a classification report
4. Interpret the results

**Code template:**
```python

# Calculate metrics
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_confusion = confusion_matrix(y_test, y_test_pred)

print("=== Model Performance ===")
print(f"\nTraining Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"\nTest Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")

print("\n=== Confusion Matrix ===")
print("                Predicted")
print("              Benign  Malignant")
print(f"Actual Benign    {test_confusion[0,0]:4d}      {test_confusion[0,1]:4d}")
print(f"      Malignant  {test_confusion[1,0]:4d}      {test_confusion[1,1]:4d}")

print("\n=== Classification Report ===")
print(classification_report(y_test, y_test_pred, target_names=cancer_data.target_names))

```

**Expected outcome:** You should achieve accuracy around 90-95%. The confusion matrix shows where the model makes mistakes.

**Checkpoint:** Verify that all metrics are calculated correctly and you understand what each metric means:
- **Accuracy:** Overall percentage of correct predictions
- **Precision:** When we predict malignant, how often are we right?
- **Recall:** Of all malignant cases, how many did we catch?

---

### Step 7: Experimenting with Different K Values

**Objective:** Understand how the number of neighbors (K) affects model performance.

**What to do:**
1. Try different values of K (1, 3, 5, 7, 9, 11)
2. Train models with each K value
3. Compare their performance
4. Choose the best K

**Code template:**
```python
# Experiment with different K values
print("\n" + "="*50)
print("EXPERIMENTING WITH DIFFERENT K VALUES")
print("="*50)

k_values = [1, 3, 5, 7, 9, 11]
results = []

for k in k_values:
    # Create and train model
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train, y_train)
    
    # Make predictions
    y_pred_temp = knn_temp.predict(X_test)
    
    # Calculate accuracy
    acc = accuracy_score(y_test, y_pred_temp)
    prec = precision_score(y_test, y_pred_temp)
    rec = recall_score(y_test, y_pred_temp)
    
    results.append({
        'K': k,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec
    })
    
    print(f"K={k:2d}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}")

# Find best K
results_df = pd.DataFrame(results)
best_k = results_df.loc[results_df['Accuracy'].idxmax(), 'K']
print(f"\nBest K value: {best_k} (Accuracy: {results_df['Accuracy'].max():.4f})")

# Visualize results
plt.figure(figsize=(10, 6))
plt.plot(results_df['K'], results_df['Accuracy'], marker='o', label='Accuracy')
plt.plot(results_df['K'], results_df['Precision'], marker='s', label='Precision')
plt.plot(results_df['K'], results_df['Recall'], marker='^', label='Recall')
plt.xlabel('K (Number of Neighbors)')
plt.ylabel('Score')
plt.title('KNN Performance vs K Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('knn_k_comparison.png', dpi=150, bbox_inches='tight')
print("\nSaved visualization to 'knn_k_comparison.png'")
plt.show()
```

**Expected outcome:** You should see that different K values give different performance. Usually, K=5 or K=7 works well, but it depends on the data.

**Checkpoint:** Verify that you can see how K affects performance and understand why.

---

## Part 2: Unguided Exercise - Customer Churn Prediction

### Background

Now it's your turn! You'll work independently to predict customer churn for a telecommunications company. Customer churn is when customers stop using a company's services. Predicting churn helps companies take proactive measures to retain customers.

**Dataset:** Telco Customer Churn Dataset
- **Source:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Alternative Download:** The dataset is also available on various data science platforms

**Dataset Description:**
- Contains information about 7,043 customers
- Features include: customer demographics, services subscribed, account information
- Target: Whether the customer churned (Yes/No)

---

### Your Task

Follow these steps to build a churn prediction model. **You will write the code yourself** - use Part 1 as a reference, but implement everything independently.

#### Step 1: Load and Explore the Data
- Download the Telco Customer Churn dataset
- Load it into a pandas DataFrame
- Explore the dataset: shape, columns, data types, missing values
- Check the target distribution (how many churned vs. didn't churn)
- Visualize some key features

#### Step 2: Data Preprocessing
- Handle missing values (if any)
- Convert categorical variables to numeric (use encoding techniques)
- Select relevant features for modeling
- Separate features (X) from target (y)
- Convert target to binary (0/1) if needed

#### Step 3: Split the Data
- Split into training (80%) and testing (20%) sets
- Use `stratify` to maintain class distribution
- Set `random_state=42` for reproducibility

#### Step 4: Train a KNN Model
- Create a KNN classifier
- Start with `n_neighbors=5`
- Train the model on the training data

#### Step 5: Make Predictions and Evaluate
- Make predictions on the test set
- Calculate accuracy, precision, and recall
- Create a confusion matrix
- Generate a classification report
- Interpret the results

#### Step 6: Experiment and Improve
- Try different K values (1, 3, 5, 7, 9, 11, 15)
- Compare their performance
- Choose the best K value
- Document your findings

#### Step 7: Analysis and Recommendations
- What is the model's accuracy?
- What features seem most important? (You can explore this by looking at feature distributions)
- What would you recommend to the company based on your model?
- What are the limitations of your model?

---

### Hints and Tips

1. **Data Loading:** The dataset is a CSV file. Use `pd.read_csv()` to load it.

2. **Categorical Encoding:** The dataset has categorical columns (like 'gender', 'Partner', 'InternetService', etc.). You'll need to convert these to numbers. Options:
   - Use `pd.get_dummies()` for one-hot encoding
   - Use `LabelEncoder` from sklearn
   - Manual mapping for binary categories

3. **Target Variable:** The 'Churn' column likely contains 'Yes'/'No' or similar. Convert to 0/1.

4. **Feature Selection:** Not all columns may be useful. Consider:
   - Dropping customer ID (if present)
   - Dropping columns with too many missing values
   - Keeping numeric and encoded categorical features

5. **Handling Missing Values:** Check for missing values. Common approaches:
   - Drop rows with missing values
   - Fill with median/mean for numeric
   - Fill with mode for categorical

6. **Evaluation Focus:** For churn prediction:
   - **Recall** is often important (catching churners)
   - **Precision** helps avoid false alarms
   - Balance depends on business needs

---

### Expected Deliverables

Create a Python script `churn_prediction.py` that includes:

1. **Data Loading and Exploration**
   - Code to load the dataset
   - Summary statistics and visualizations
   - Documentation of findings

2. **Data Preprocessing**
   - Code to handle missing values
   - Code to encode categorical variables
   - Feature selection

3. **Model Training and Evaluation**
   - Code to train KNN model
   - Code to evaluate performance
   - Results with different K values

4. **Analysis and Conclusions**
   - Written summary of findings
   - Model performance metrics
   - Business recommendations

---

## Submission Guidelines

### What to Submit

**Part 1 (Guided Exercise):**
- [ ] Your complete `breast_cancer_prediction.py` file
- [ ] Screenshots or output showing:
  - Dataset exploration
  - Model training
  - Evaluation metrics
  - K value comparison results

**Part 2 (Unguided Exercise):**
- [ ] Your complete `churn_prediction.py` file
- [ ] Screenshots or output showing:
  - Data exploration and preprocessing
  - Model training and evaluation
  - Comparison of different K values
- [ ] A brief report (1-2 pages) including:
  - Summary of your approach
  - Model performance metrics
  - Key findings about customer churn
  - Business recommendations
  - Limitations and future improvements

### How to Submit

**Upload your work:**
1. **For code files:** Upload both Python files
2. **For screenshots:** Upload PNG or JPG images
3. **For reports:** Upload PDF or text file

**Important notes:**
- Make sure all code runs without errors
- Include comments explaining your approach
- Document any challenges you faced
- Show your thought process in the code comments

**Due date:** Check with your instructor for the specific due date.

---

## Troubleshooting

**Common issues and solutions:**

**Issue 1: "ValueError: Input contains NaN"**
- **Solution:** Handle missing values before training. Use `df.dropna()` or `df.fillna()` methods.

**Issue 2: "ValueError: Unknown label type: 'object'"**
- **Solution:** Your target variable is still a string. Convert it to numeric using encoding or mapping.

**Issue 3: "TypeError: '<' not supported between instances of 'str' and 'float'"**
- **Solution:** Some columns are strings but should be numeric. Check data types with `df.dtypes` and convert as needed.

**Issue 4: "Low accuracy on churn prediction"**
- **Solution:** 
  - Try different K values
  - Check if you need to scale features (KNN is sensitive to scale)
  - Review feature selection - maybe some features aren't useful
  - Check class imbalance - if most customers don't churn, accuracy might be misleading

**Issue 5: "Confusion about encoding categorical variables"**
- **Solution:** For binary categories (Yes/No), use simple mapping: `df['column'] = df['column'].map({'Yes': 1, 'No': 0})`. For multiple categories, use `pd.get_dummies()`.

**Issue 6: "Model performs poorly"**
- **Solution:** 
  - KNN works better with scaled features. Try using `StandardScaler` from sklearn
  - Experiment with different K values
  - Consider feature engineering (create new features from existing ones)

---

## Bonus Challenges

If you finish early, try these additional challenges:

### Challenge 1: Feature Scaling
KNN is sensitive to the scale of features. Implement feature scaling:
- Use `StandardScaler` from `sklearn.preprocessing`
- Scale both training and test sets
- Compare performance before and after scaling
- Document the improvement

### Challenge 2: Feature Importance Analysis
Analyze which features are most important:
- Try removing features one by one and see how it affects performance
- Use correlation analysis to find features most related to churn
- Create visualizations showing feature importance

### Challenge 3: Other Algorithms
Try other sklearn classifiers:
- `LogisticRegression`
- `RandomForestClassifier`
- `SVC` (Support Vector Classifier)
- Compare their performance with KNN

### Challenge 4: Advanced Evaluation
Implement additional evaluation techniques:
- ROC curve and AUC score
- Precision-Recall curve
- Cross-validation for more robust evaluation
- Learning curves to understand model behavior

### Challenge 5: Business Impact Analysis
Create a business-focused analysis:
- Calculate the cost of false negatives (missed churners)
- Calculate the cost of false positives (unnecessary retention efforts)
- Find the optimal threshold for churn prediction
- Create actionable recommendations for the business

---

## Additional Resources

**Reference materials:**
- [sklearn KNN Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- [sklearn Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- [Understanding Confusion Matrix](https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/)

**Tools and software:**
- [sklearn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Tutorial](https://matplotlib.org/stable/tutorials/index.html)

**Further reading:**
- [Introduction to Machine Learning](https://scikit-learn.org/stable/getting_started.html)
- [KNN Algorithm Explained](https://www.analyticsvidhya.com/blog/2018/03/introduction-k-neighbours-algorithm-clustering/)
- [Customer Churn Prediction Best Practices](https://www.kaggle.com/learn/customer-churn-prediction)

---

## To-Do List

**Part 1 (Guided):**
- [ ] Complete Step 1: Setting Up and Loading Data
- [ ] Complete Step 2: Data Exploration
- [ ] Complete Step 3: Splitting the Data
- [ ] Complete Step 4: Training the KNN Model
- [ ] Complete Step 5: Making Predictions
- [ ] Complete Step 6: Evaluating Model Performance
- [ ] Complete Step 7: Experimenting with Different K Values

**Part 2 (Unguided):**
- [ ] Download and load the Telco Customer Churn dataset
- [ ] Explore and preprocess the data
- [ ] Train a KNN model
- [ ] Evaluate and compare different K values
- [ ] Write analysis and recommendations
- [ ] Submit your work

---

## Learning Reflection

After completing this lab, reflect on what you've learned:

1. **Machine Learning Workflow:** What are the essential steps in building an ML model?
2. **Train/Test Split:** Why is it important to separate training and testing data?
3. **KNN Algorithm:** How does KNN make predictions? What are its strengths and weaknesses?
4. **Evaluation Metrics:** What do accuracy, precision, and recall tell you? When is each most important?
5. **Data Preprocessing:** What challenges did you face with the churn dataset? How did you solve them?
6. **Model Selection:** How did you choose the best K value? What factors influenced your decision?

**Key takeaways:**
- Machine learning follows a consistent workflow: load → explore → preprocess → split → train → evaluate
- Evaluation metrics help you understand model performance from different perspectives
- Data preprocessing is often the most time-consuming but crucial step
- Experimentation and iteration are key to improving model performance
- Understanding your data and problem domain is essential for building effective models

These skills form the foundation for more advanced machine learning and AI applications.

---

*Good luck with your machine learning journey! Remember to experiment, document your findings, and don't hesitate to ask questions when you get stuck.*
