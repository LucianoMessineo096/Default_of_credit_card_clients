import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np

class Perceptron(nn.Module):

    def __init__(self,input_size):

        super().__init__()

        self.fc = nn.Linear(input_size,1)
        self.sigmoid = nn.Sigmoid()
        self.losses= []
        self.y_train_preds = []
        self.y_test_preds = []

    def forward(self , x):

        x = self.fc(x)
        x = self.sigmoid(x)

        return x

    def train_phase(self,X_train,y_train,epochs,lr):

        criterion = nn.BCELoss()
        optimizer = torch.optim.SGD(self.parameters(),lr=lr)

        for epoch in range(epochs):

            optimizer.zero_grad()
            y_pred = self(X_train)
            loss = criterion(y_pred,y_train.unsqueeze(1))
            loss.backward()
            optimizer.step()

            self.losses.append(loss.item())
            if epoch == 1 :
                self.y_train_preds.append(y_pred)


    def test_phase(self,X_test,y_test):

        with torch.no_grad():

            y_pred = self(X_test)

            self.y_test_preds.append(y_pred)

    def plot_loss_fn(self):

        plt.plot(self.losses, label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss over Epochs")
        plt.show()

    def get_eval_metrics(self,y,y_pred):

        '''print(y_pred)
        print("-----------------------------------------------------")'''
        y_pred = torch.cat(y_pred,dim=0).detach().numpy().flatten()
        '''print(y_pred)
        print("-----------------------------------------------------")'''
        y_pred = (y_pred >= 0.5).astype(float)
        '''print(y_pred)
        print("-----------------------------------------------------")'''

        y = np.array(y).flatten()
        '''print(y)'''

        accuracy = accuracy_score(y,y_pred) 
        precision = precision_score(y,y_pred)    
        recall = recall_score(y,y_pred)   
        f1 = f1_score(y,y_pred)
        matrix =confusion_matrix(y, y_pred)

        return accuracy,precision,recall,f1,matrix

        '''metrics = pd.DataFrame({
            'accuracy' : accuracy_score(y,y_pred),
            'precision' : precision_score(y,y_pred),
            'recall' : recall_score(y,y_pred),
            'f1': f1_score(y,y_pred),
            
        })
       

        return metrics'''
    
    