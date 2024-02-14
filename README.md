# Default_of_credit_card_clients

@Luciano Messineo
@University of Palermo
@Artificial Intelligence


There is both python code within the code folder and a Jupyter notebook for better reading of the results.

# Dataset Description

The dataset contains features related to the economic and demographic conditions of bank customers.

**Dataset characteristics:** Multivariate
**Subject Area:** Business
**Task:** Classification
**Feature type:** Integer,Real
**Instances:** 30k
**Features:** 23

**Variable information**

This study utilized a binary variable, default payment (Yes = 1, No = 0), as the response variable. A comprehensive literature review guided the selection of 23 explanatory variables, each serving as a potential predictor. The variables are as follows:

X1: Amount of the given credit (NT dollar): This encompasses both individual consumer credit and supplementary family credit.
X2: Gender (1 = male; 2 = female).
X3: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others).
X4: Marital status (1 = married; 2 = single; 3 = others).
X5: Age (year).
X6 - X11: History of past payment. Monthly payment records were tracked from April to September 2005, with X6 representing the repayment status in September 2005, X7 in August 2005, and so forth. The measurement scale for the repayment status includes values such as -1 for paying duly, 1 for payment delay for one month, 2 for payment delay for two months, and up to 9 for payment delay for nine months and above.
X12-X17: Amount of bill statement (NT dollar). X12 represents the amount of the bill statement in September 2005, X13 in August 2005, and so on up to X17 in April 2005.
X18-X23: Amount of previous payment (NT dollar). X18 denotes the amount paid in September 2005, X19 in August 2005, and continuing up to X23 in April 2005.

# Project

The goal of this project is to predict if a customer can repay his credit card debts next month. For this classification task, we will utilize three approaches:

1. Decision Tree
2. Single-layer perceptron
3. Deep neural network

and compare their efficiency.

### How can we compare the efficiency of the three approaches?

We will utilize the following metrics:

1. **Accuracy:**
   Provides the percentage of correct predictions compared to the total predictions made.
2. **Precision:**
   Measures the percentage of instances predicted as positive that are truly positive.
   \[ \text{Precision} = \frac{\text{True positives}}{\text{True positives} + \text{False positives}} \]
3. **Recall:**
   Measures the percentage of positive instances that were predicted correctly.
4. **F1-Score:**
   The harmonic mean between precision and recall.
   \[ \text{F1} = \frac{2 \cdot (\text{Precision} \cdot \text{Recall})}{\text{Precision} + \text{Recall}} \]
5. **Confusion Matrix:**
   Shows the number of correct predictions and errors made for each class.
