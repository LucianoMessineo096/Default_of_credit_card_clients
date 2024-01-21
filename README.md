# Default_of_credit_card_clients

@Luciano Messineo
@University of Palermo
@Artificial Intelligence

# Dataset Description

The dataset contains features related to the economic and demographic conditions of bank customers.

# Project

The goal of this project is to predict if a customer can repay his credit card debts next month. For this classification task, we will utilize three approaches:

1. Decision Tree
2. Single-layer perceptron
3. Deep neural network

and compare their efficiency.

# First Approach (Decision Tree)

This is a supervised learning technique that leverages decision rules directly inferred from data. The tree receives an input vector and performs a series of tests to determine its class membership. These tests typically involve binary decisions of the form "x < a?". The goal is to achieve successive divisions of the feature space, creating regions corresponding to different classes.

How does it work?

At each division step, the following actions are necessary:

1. **Determine the Threshold Value:**
   Given a feature, calculate the midpoints between two consecutive values.
   \[ \text{threshold} = \frac{A + B}{2} \]
   Once all midpoints are calculated, choose the one that generates the most homogeneous descendants, and here comes the next point, i.e., determining the most appropriate split criterion.

2. **Choose the Feature for the Next Split:**
   For choosing the criterion, we can opt for:
   a. Gini Index
   b. Shannon Entropy
   Therefore, we will choose the point that maximizes the decrease in impurity.
   \[ \Delta I = I_{\text{parent}} - \frac{N_1}{N} \cdot I_1 - \frac{N_2}{N} \cdot I_2 \]
   where:
   \( I_{\text{parent}} \) represents the impurity of the distribution of classes in the node before the split, calculated using one of the previous split criteria.
   \( N_1 \) and \( N_2 \) represent the number of occurrences below and above the threshold value, respectively.
   \( N \) is the total number of occurrences.
   \( I_1 \) and \( I_2 \) represent the values provided by the split criterion for the left and right tree nodes, respectively.

3. **Determine the Stopping Criterion:**
   We can choose different criteria, such as the maximum tree depth.

4. **Determine the Class for a Leaf Node:**
   To determine the class of a leaf node, start by counting the number of occurrences for each class. The dominant class is the one with the highest count.

To implement the following approach, we will use the sklearn library and compare a decision tree using the Gini criterion with one using Shannon's entropy.



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
