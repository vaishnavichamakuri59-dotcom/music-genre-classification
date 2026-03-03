import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

class MusicGenreClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MusicGenreClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_and_save_model():
    dataset = pd.read_csv("combined_music_dataset.csv")

    # Use only MFCC columns
    X = dataset.drop(columns=['filename', 'genre', 'duration(in sec)'])
    y = dataset['genre']

    # Encode labels properly
    labels = y.unique()
    label_mapping = {label: idx for idx, label in enumerate(labels)}
    reverse_mapping = {idx: label for label, idx in label_mapping.items()}

    y_encoded = y.map(label_mapping)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.3, random_state=42
    )

    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train.values)

    model = MusicGenreClassifier(X_train.shape[1], len(labels))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    #  ADD EVALUATION HERE
    model.eval()

    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test.values)

    with torch.no_grad():
        outputs = model(X_test_tensor)
        predicted = torch.argmax(outputs, dim=1)
        accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)

    print(f"Test Accuracy: {accuracy:.4f}")

    # Save everything
    torch.save(model.state_dict(), "model/model.pth")
    joblib.dump(scaler, "model/scaler.pkl")
    joblib.dump(label_mapping, "model/label_mapping.pkl")

    print("Model Trained and Saved Successfully!")


if __name__ == "__main__":
    train_and_save_model()