import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import os
import pandas as pd


class DeepNeuralNetwork(nn.Module):

    def __init__(self,input_size,hidden_size1,hidden_size2,hidden_size3,hidden_size4,output_size,lr,epochs,weight_decay):

        super().__init__()

        self.input_size=input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
        self.hidden_size4 = hidden_size4
        self.output_size = output_size
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay

        self.criterion = None
        self.optimizer = None

        self.model = nn.Sequential(

            nn.Linear(input_size,hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1,hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2,hidden_size3),
            nn.ReLU(),
            nn.Linear(hidden_size3,hidden_size4),
            nn.ReLU(),
            nn.Linear(hidden_size4,output_size),
            nn.Sigmoid()

        )

        self.losses = []
        self.y_train_preds = []
        self.y_test_preds = []


    def forward(self,x):

        return self.model(x)

    def training_phase(self,X_train,y_train):

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(),lr=self.lr,weight_decay=self.weight_decay)

        for epoch in range(self.epochs):

            optimizer.zero_grad()
            y_pred = self(X_train)
            loss = criterion(y_pred,y_train.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            if epoch % 100 == 0:
                print(f"Epoch = {epoch} - Loss = {loss.item()} ")

            self.losses.append(loss.item())
            if epoch == self.epochs-1:
                self.y_train_preds.append(y_pred)

        self.criterion = criterion
        self.optimizer = optimizer


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
        
        y_pred = torch.cat(y_pred,dim=0).detach().numpy().flatten()
        y_pred = (y_pred >= 0.5).astype(float)
        y = np.array(y).flatten()
        

        accuracy = accuracy_score(y,y_pred) 
        precision = precision_score(y,y_pred)    
        recall = recall_score(y,y_pred)   
        f1 = f1_score(y,y_pred)
        matrix =confusion_matrix(y, y_pred)

        return pd.DataFrame({
            'accuracy': [accuracy],
            'precision': [precision],
            'recall': [recall],
            'f1': [f1],
            'matrix': [matrix]  
        })

    def save_model(self,model,train_metrics):

        dir = '../saved_model'

        if not os.path.exists(dir):
            os.makedirs(dir)

        existing_models = [file for file in os.listdir(dir) if "dnn" in file]
        count = len(existing_models)

        new_model_name = f"dnn_{count+1}.pth"

        torch.save(

            {
                'model_state_dict': model.state_dict(),
                'epochs' : self.epochs,
                'lr' : self.lr,
                'optimizer': self.optimizer.state_dict(),
                'criterion': self.criterion,
                'weight_decay': self.weight_decay,
                'structure':{

                    'n_hidden_layers': 4,
                    'n_neuron_input_layer':self.input_size,
                    'n_neuron_hidden_layer1':self.hidden_size1,
                    'n_neuron_hidden_layer2':self.hidden_size2,
                    'n_neuron_hidden_layer3':self.hidden_size3,
                    'n_neuron_hidden_layer4':self.hidden_size4,
                    'out':1
                },
                'train_metrics': train_metrics,
                'losses' :self.losses,
            },
            os.path.join(dir,new_model_name)
        )
