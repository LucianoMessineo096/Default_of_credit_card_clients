# Default_of_credit_card_clients

@Luciano Messineo
@Università degli studi di Palermo
@Intelligenza Artificiale

# Dataset description

The dataset contains features related to the economic and demographic conditions of bank customers.


# Project

The goal of this project is to predict if a customer can repay his credit card debts next month. For this classification task, we will utilize three approaches:

    1. Decision Tree
    2. Single layer perceptron
    3. Deep neural network

and compare their efficency.



### how we can compare the efficency of the three approaches ?

WWe will utilize the following metrics:

1. Accuracy

    Provides the percentage of correct predictions compared to the total predictions made.

2. Precision

    Measures the percentage of instances predicted as positive that are truly positive.
    Precision = True_positives / (True_positives + False_positives)
    
3. Recall

    Measures the percentage of positive instances that were predicted correctly.
    
4. F1-Score:

    The harmonic mean between precision and recall.
    F1 = 2 * (Precision * Recall) / (Precision + Recall)

5. Confusion Matrix:

    Shows the number of correct predictions and errors made for each class.