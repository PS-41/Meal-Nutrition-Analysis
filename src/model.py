import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalModel(nn.Module):
    def __init__(self, image_size=(64, 64, 3), cgm_size=16, demo_size=31, 
                 dropout_rate=0.5, cnn_filters=(16, 32), lstm_hidden_size=64, num_lstm_layers=1):
        super().__init__()

        # Image branch (CNN)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, cnn_filters[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(cnn_filters[0], cnn_filters[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(cnn_filters[1] * (image_size[0] // 4) * (image_size[1] // 4), 128),
            nn.ReLU()
        )

        # Time-series branch (LSTM)
        self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_hidden_size, num_layers=num_lstm_layers, batch_first=True)
        self.lstm_fc = nn.Linear(lstm_hidden_size * cgm_size, 128)

        # Demographics branch (Feed-forward NN)
        self.demo_fc = nn.Sequential(
            nn.Linear(demo_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Dropout added here
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)  # Dropout added here
        )

        # Joint embedding and prediction
        self.fc = nn.Sequential(
            nn.Linear(128 * 3, 128),  # Concatenate three branches
            nn.ReLU(),
            nn.Linear(128, 1)  # Output: Lunch Calories
        )

    def forward(self, image_breakfast, image_lunch, cgm_data, demo_data):
        # Image branch (process both breakfast and lunch images)
        img_b = self.cnn(image_breakfast)
        img_l = self.cnn(image_lunch)
        img_features = img_b + img_l  # Combine image features (can experiment with concatenation later)

        # Time-series branch
        cgm_data = cgm_data.unsqueeze(-1)  # Add channel dimension for LSTM
        lstm_out, _ = self.lstm(cgm_data)
        lstm_features = self.lstm_fc(lstm_out.reshape(lstm_out.size(0), -1))

        # Demographics branch
        demo_features = self.demo_fc(demo_data)

        # Concatenate all features
        joint_embedding = torch.cat([img_features, lstm_features, demo_features], dim=1)

        # Final prediction
        output = self.fc(joint_embedding)
        return output
