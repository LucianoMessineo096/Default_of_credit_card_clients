from utils import import_data,show_graphs,split_dataset
from DecisionTree import DecisionTree
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
from Perceptron import Perceptron
import torch

def decision_tree_approach(X_train, X_test, y_train, y_test):

    depths_to_test = [3,5,7,10]
    criterion="gini"
    splitter="best"
    fold_number=10

    dt_agent = DecisionTree(
        criterion=criterion,
        split_criterion=splitter,
        depths_to_test=depths_to_test,
        fold_number=fold_number,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )    
    
    dt_agent.execute_k_fold_cross_validation()
    cv_metrics = dt_agent.get_cv_metrics()
    


    #create the new model based on the best depth 

    best_depth=5

    best_dt = tree.DecisionTreeClassifier(
        criterion=criterion,
        splitter=splitter,
        max_depth=best_depth
    )

    #train the new model
    best_dt.fit(X_train,y_train)
    y_train_pred = best_dt.predict(X_train)

    #test the new model
    y_pred = best_dt.predict(X_test)

    #get train and test metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    train_conf_matrix = confusion_matrix(y_train, y_train_pred)

    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    test_conf_matrix = confusion_matrix(y_test, y_pred)

    print("------------------------------------------------------------------------------")

    print("Criterion: " + str(criterion))
    print("Splitter: "+ str(splitter))
    print("Fold number: "+ str(fold_number))

    print("------------------------------------------------------------------------------")
    print("Metriche di cross validation")
    print(cv_metrics)

    print("------------------------------------------------------------------------------")
    print("Metriche sul training set:")
    train_metrics = pd.DataFrame({
        'Profondità': best_depth,
        'Accuratezza': [train_accuracy],
        'Precisione': [train_precision],
        'Recall': [train_recall],
        'F1-Score': [train_f1],
        'Confusion Matrix': [train_conf_matrix],
    })
    
    print(train_metrics)

    print("------------------------------------------------------------------------------")
    print("Metriche sul test set:")
    test_metrics  =pd.DataFrame({
        'Profondità': best_depth,
        'Accuratezza': [test_accuracy],
        'Precisione': [test_precision],
        'Recall': [test_recall],
        'F1-Score': [test_f1],
        'Confusion Matrix': [test_conf_matrix],
    })
    print(test_metrics)

def perceptron_approach(X_train,X_test,y_train,y_test):

    X_train_tensor = torch.tensor(X_train.values,dtype=torch.float32)
    #print(X_train_tensor.shape)
    X_test_tensor = torch.tensor(X_test.values,dtype=torch.float32)
    #print(X_test_tensor.shape)
    y_train_tensor = torch.tensor(y_train.values,dtype=torch.float32)
    #print(y_train_tensor.shape)
    y_test_tensor = torch.tensor(y_test.values,dtype=torch.float32)
    #print(y_test_tensor.shape)

    input_size = X_train.shape[1]
    #print(input_size)
    epochs = 100
    lr=0.01

    perceptron = Perceptron(input_size=input_size)

    perceptron.train_phase(X_train_tensor,y_train_tensor,epochs,lr)

    perceptron.test_phase(X_test_tensor,y_test_tensor)

    train_metrics = perceptron.get_eval_metrics(y_train,perceptron.y_train_preds)
    test_metrics = perceptron.get_eval_metrics(y_test,perceptron.y_test_preds)

    print("-------------------------------------------------")
    print("Train metrics")
    print(train_metrics)

    print("--------------------------------------------------")
    print("Test metrics")
    print(test_metrics)

    perceptron.plot_loss_fn()
    

def main():

    data = import_data()
    #show_graphs(data)

    X_train, X_test, y_train, y_test = split_dataset(data,0.2,42)
    
    #decision_tree_approach(X_train, X_test, y_train, y_test)

    perceptron_approach(X_train,X_test,y_train,y_test)


    

main()