import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd

class DeepNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(DeepNeuralNetwork, self).__init__()
        layers = []
        for i in range(len(hidden_sizes)):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_sizes[i]))
            else:
                layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.model(x))

def train_model(model, X_train, y_train, epochs, lr):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output.squeeze(), y_train)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

def evaluate_model(model, X_test, y_test):
    with torch.no_grad():
        output = model(X_test)
        predictions = (output > 0.5).float()
        accuracy = (predictions == y_test).float().mean()
        print(f"Accuracy: {accuracy.item()}")

def prepare_data():
    # Carica il DataFrame senza interpretare la prima riga come intestazione e specifica i nomi delle colonne
    data = pd.read_csv('../assets/default_of_credit_card_clients.csv', header=None)

    print(data.dtypes)


    # Seleziona le feature (tutte le colonne tranne 'default payment next month')
    X = data.iloc[:, :-1]

    # Seleziona il target (colonna 'default payment next month')
    y = data['default_payment_next_month']

    # Dividi il dataset in set di addestramento e test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def build_and_train_model():
    X_train, X_test, y_train, y_test = prepare_data()

    # Converti colonne numeriche in tipi di dati numerici
    numeric_columns = ['LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 
                    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    X_train[numeric_columns] = X_train[numeric_columns].astype(float)
    X_test[numeric_columns] = X_test[numeric_columns].astype(float)


    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    input_size = X_train.shape[1]
    hidden_sizes = [64, 32]  # Esempio di architettura della rete
    output_size = 1

    model = DeepNeuralNetwork(input_size, hidden_sizes, output_size)

    # Trained model
    train_model(model, X_train_tensor, y_train_tensor, epochs=1000, lr=0.001)

    # Valutazione del modello
    evaluate_model(model, torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

if __name__ == "__main__":
    build_and_train_model()
