# Default_of_credit_card_clients

@Luciano Messineo
@Università degli studi di Palermo
@Intelligenza Artificiale

# Dataset description

The dataset contains features related to economical and demographical condition of a bank customers.


# Project

The aim of this project is to predict if a customer can repay his credit card debts next month.
To do this classification task we will utilize three approaches:

    1. Decision Tree
    2. Single layer perceptron
    3. Deep neural network

and compare their efficency.



### how we can compare the efficency of the three approaches ?

We will utilize this metrics:

    1. Accuracy

        fornisce la percentuale di previsioni corrette rispetto al totale delle previsioni effettuate

    2. Precision

        misura la percentuale di istanze predette come positive che sono realmente positive 

        Precision = True_positives / (True_positives + False_positives)

    3. Recall

        misura la percentuale di istanze positive che sono state predette correttamente

    3. F1-score

        si tratta della media armonica tra la precision e la recall

        F1 = 2 * (Precision * Recall)/(Precision+Recall)

    4. Matrice di confusione

        Mostra il numero di predizioni corrette e gli errori commessi per ciascuna classe