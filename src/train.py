import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# Define the hierarchical ResNet model
class HierarchicalResNet(nn.Module):
    def __init__(self, num_master_classes, num_sub_classes, num_specific_classes):
        super(HierarchicalResNet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove the original fully connected layer
        
        # Add new fully connected layers for each level of the hierarchy
        self.master_fc = nn.Linear(self.resnet.fc.in_features, num_master_classes)
        self.sub_fc = nn.Linear(self.resnet.fc.in_features, num_sub_classes)
        self.specific_fc = nn.Linear(self.resnet.fc.in_features, num_specific_classes)
    
    def forward(self, x):
        x = self.resnet(x)
        master_out = self.master_fc(x)
        sub_out = self.sub_fc(x)
        specific_out = self.specific_fc(x)
        return master_out, sub_out, specific_out

# Function to create hierarchical labels
def create_hierarchical_labels(dataset_dir, label_file='labels.json'):
    if os.path.exists(label_file):
        with open(label_file, 'r') as f:
            labels = json.load(f)
        return labels['master_labels'], labels['sub_labels'], labels['specific_labels'], labels['master_to_idx'], labels['sub_to_idx'], labels['specific_to_idx']
    
    master_classes = sorted(os.listdir(dataset_dir))
    sub_classes = []
    specific_classes = []
    
    master_to_idx = {cls: idx for idx, cls in enumerate(master_classes)}
    sub_to_idx = {}
    specific_to_idx = {}
    
    master_labels = []
    sub_labels = []
    specific_labels = []
    
    for master_class in master_classes:
        master_path = os.path.join(dataset_dir, master_class)
        sub_classes = sorted(os.listdir(master_path))
        
        for sub_class in sub_classes:
            sub_path = os.path.join(master_path, sub_class)
            specific_classes = sorted(os.listdir(sub_path))
            
            for specific_class in specific_classes:
                specific_path = os.path.join(sub_path, specific_class)
                for img_name in os.listdir(specific_path):
                    if sub_class not in sub_to_idx:
                        sub_to_idx[sub_class] = len(sub_to_idx)
                    if specific_class not in specific_to_idx:
                        specific_to_idx[specific_class] = len(specific_to_idx)
                    
                    master_labels.append(master_to_idx[master_class])
                    sub_labels.append(sub_to_idx[sub_class])
                    specific_labels.append(specific_to_idx[specific_class])
    
    labels = {
        'master_labels': master_labels,
        'sub_labels': sub_labels,
        'specific_labels': specific_labels,
        'master_to_idx': master_to_idx,
        'sub_to_idx': sub_to_idx,
        'specific_to_idx': specific_to_idx
    }
    
    with open(label_file, 'w') as f:
        json.dump(labels, f)
    
    return master_labels, sub_labels, specific_labels, master_to_idx, sub_to_idx, specific_to_idx

# Load dataset and create DataLoader
def load_dataset(dataset_dir, batch_size=32, train_size=0.7, val_size=0.15, test_size=0.15):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    dataset = datasets.ImageFolder(dataset_dir, transform=transform)
    
    # Calculate the number of samples for each split
    train_len = int(train_size * len(dataset))
    val_len = int(val_size * len(dataset))
    test_len = len(dataset) - train_len - val_len
    
    # Split dataset into train, validation, and test sets
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_len, val_len, test_len])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def main():
    dataset_dir = 'path/to/your/dataset'

    num_epochs = 10
    batch_size=32
    learning_rate=0.001

    master_labels, sub_labels, specific_labels, master_to_idx, sub_to_idx, specific_to_idx = create_hierarchical_labels(dataset_dir)
    
    num_master_classes = len(master_to_idx)
    num_sub_classes = len(sub_to_idx)
    num_specific_classes = len(specific_to_idx)
    
    model = HierarchicalResNet(num_master_classes, num_sub_classes, num_specific_classes)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_size=0.7
    val_size=0.15
    test_size=0.15

    train_loader, val_loader, _ = load_dataset(dataset_dir, batch_size, train_size, val_size, test_size)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, _ in train_loader:
            optimizer.zero_grad()
            
            master_out, sub_out, specific_out = model(inputs)
            
            master_loss = criterion(master_out, torch.tensor(master_labels))
            sub_loss = criterion(sub_out, torch.tensor(sub_labels))
            specific_loss = criterion(specific_out, torch.tensor(specific_labels))
            
            loss = master_loss + sub_loss + specific_loss
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        train_losses.append(running_loss / len(train_loader))
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, _ in val_loader:
                master_out, sub_out, specific_out = model(inputs)
                
                master_loss = criterion(master_out, torch.tensor(master_labels))
                sub_loss = criterion(sub_out, torch.tensor(sub_labels))
                specific_loss = criterion(specific_out, torch.tensor(specific_labels))
                
                loss = master_loss + sub_loss + specific_loss
                val_loss += loss.item()
        
        val_losses.append(val_loss / len(val_loader))
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]}, Val Loss: {val_losses[-1]}")
    
    print("Training complete")
    
    # Plot the training and validation loss
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
