from utils import import_data,show_graphs,get_tensors,preprocessing,test_models,show_heatmap
from DecisionTree import DecisionTree
from DeepNeuralNetwork import DeepNeuralNetwork
import pandas as pd
from sklearn.tree import DecisionTreeClassifier , plot_tree
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
from sklearn.linear_model import Perceptron
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch

def deep_neural_network(X_train,X_test,y_train,y_test):

    print("----------------Deep Neural Network----------------------")

    X_train_tensor,X_test_tensor,y_train_tensor,y_test_tensor = get_tensors(X_train,X_test,y_train,y_test)

    input_size = X_train.shape[1] 
    hidden_sizes = [32,64,128,64]
    output_size = 1

    epochs = 1000
    lr = 0.005
    weight_decay = 0.005 

    dnn = DeepNeuralNetwork(
        input_size=input_size,
        hidden_sizes = hidden_sizes,
        output_size=output_size,
        lr=lr,
        epochs=epochs,
        weight_decay = weight_decay
    )

    dnn.training_phase(X_train_tensor,y_train_tensor)

    dnn.test_phase(X_test_tensor,y_test_tensor)

    train_metrics = dnn.get_eval_metrics(y_train,dnn.y_train_preds)
    test_metrics = dnn.get_eval_metrics(y_test,dnn.y_test_preds)

    print("-------------------------------------------------")
    print("Train metrics")
    print(train_metrics)

    print("--------------------------------------------------")
    print("Test metrics")
    print(test_metrics)

    dnn.plot_loss_fn()

    dnn.save_model(
        dnn,
        train_metrics    
    )

def decision_tree(X_train, X_test, y_train, y_test):

    print("------------------Decision Tree-----------------------")

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

    best_dt = DecisionTreeClassifier(
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

    plt.figure(figsize=(20, 10))
    plot_tree(best_dt, feature_names=X_train.columns, class_names=[str(cls) for cls in y_train.unique()], filled=True)
    plt.show()
    
def perceptron(X_train, X_test, y_train, y_test,X,y):

    print("----------Percettrone-------------------")

    losses=[]
    train_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'conf_matrix': []
    }

    epochs = 500
    lr = 0.03

    #Model
    perceptron = Perceptron(
        max_iter=epochs,
        eta0=lr,
        random_state=42
    )

    #Training phase    

    for epoch in range(epochs):

        perceptron.partial_fit(X_train,y_train,classes=np.unique(y_train))
        y_pred_train = perceptron.predict(X_train)

        if (epoch +1 )%100==0:

            loss = np.mean(y_pred_train != y_train)
            losses.append(loss)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss}')

            train_accuracy = accuracy_score(y_train, y_pred_train) *100
            train_precision = precision_score(y_train, y_pred_train) *100
            train_recall = recall_score(y_train, y_pred_train) *100
            train_f1 = f1_score(y_train, y_pred_train) *100
            train_conf_matrix = confusion_matrix(y_train, y_pred_train)

            train_metrics['accuracy'].append(train_accuracy)
            train_metrics['precision'].append(train_precision)
            train_metrics['recall'].append(train_recall)
            train_metrics['f1'].append(train_f1)
            train_metrics['conf_matrix'].append(train_conf_matrix)

    plt.plot(range(100, epochs + 1, 100), losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Perceptron Training Loss')
    plt.legend()
    plt.show()

    
    train_metrics = pd.DataFrame(train_metrics)
    print("-------------------------------------------------")
    print("Train metrics")
    print(train_metrics)

    #Test phase
    y_pred_test = perceptron.predict(X_test)
    
    #Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred_test) *100
    precision = precision_score(y_test, y_pred_test) *100
    recall = recall_score(y_test, y_pred_test) *100
    f1 = f1_score(y_test, y_pred_test) * 100
    conf_matrix = confusion_matrix(y_test, y_pred_test) 

    print("--------------------------------------------------")
    print("Test metrics")
    print(f"accuracy: {accuracy}")
    print(f"precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")
    print("confusion matrix:")
    print(conf_matrix)

def main():

    #IMPORT

    data = import_data()

    flag = True

    while flag:
        print("Default of credits cards clients")
        print("-----------------------------------")
        print("1. Visualizzare le informazioni sul dataset")
        print("2. Visualizzare la correlazione tra le features")
        print("3. Visualizzare le prestazioni del percettrone")
        print("4. Visualizzare le prestazioni dell'albero decisionale")
        print("5. Visualizzare le prestazioni della rete neurale")
        print("6. Effettua il confronto tra le reti neurali salvate")
        print("0. Uscire dal programma")

        choice = input("Scegli un'opzione: ")

        if choice == '1':
            show_graphs(data)

        elif choice == '2':
            show_heatmap(data)

        elif choice == '3':

            #PRE-PROCESSING PHASE
            X,y = preprocessing(data)

            #SPLIT
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=123)

            perceptron(X_train,X_test,y_train,y_test,X,y)

        elif choice == '4':

            #PRE-PROCESSING PHASE
            X,y = preprocessing(data)

            #SPLIT
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=123)

            decision_tree(X_train, X_test, y_train, y_test)
            
        elif choice == '5':

            #PRE-PROCESSING PHASE
            X,y = preprocessing(data)

            #SPLIT
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=123)

            deep_neural_network(X_train,X_test,y_train,y_test)

        elif choice == '6':

            #PRE-PROCESSING PHASE
            X,y = preprocessing(data)

            #SPLIT
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=123)

            test_models(X_train,X_test,y_train,y_test)

        elif choice == '0':
            
            flag = False

        else:
            print("Opzione non valida. Riprova.")

if __name__ == "__main__":
    main()



