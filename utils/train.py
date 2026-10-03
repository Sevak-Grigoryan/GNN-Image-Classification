import os
import sys
import tqdm
import torch
import numpy as np
import torch_geometric
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Add path for module accessibility
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  
from utils.directions import *
from utils.data_processing import Image2Graph

class GNNTrainer:
    """
    A universal trainer class for Graph Neural Networks.
    
    This class handles the training loop, validation logic, learning rate 
    scheduling, and stratified dataset splitting for PyG datasets.
    """
    def __init__(self, model, device, lr=1e-3, weight_decay=5e-4):
        """
        Initializes the GNN Trainer.

        Args:
            model (torch.nn.Module): The GNN model to train.
            device (torch.device): Device to run on (cuda or cpu).
            lr (float): Initial learning rate.
            weight_decay (float): L2 regularization factor.
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-6
        )

    @staticmethod
    def split_dataset(dataset, test_size=0.15, val_size=0.15, random_state=42):
        """
        Performs a stratified split into Train, Validation, and Test sets.

        Args:
            dataset (Dataset): The full PyG dataset object.
            test_size (float): Proportion of data for the test set.
            val_size (float): Proportion of data for the validation set.
            random_state (int): Seed for reproducibility.

        Returns:
            tuple: (train_subset, val_subset, test_subset)
        """
        indices = np.arange(len(dataset))
        labels = [data.y.item() for data in dataset]

        # First split: Separate Test set
        train_idx, test_idx, train_y, _ = train_test_split(
            indices, labels, test_size=test_size, stratify=labels, random_state=random_state
        )
        
        # Second split: Separate Validation set from the remaining training data
        val_ratio_adj = val_size / (1.0 - test_size)
        train_idx, val_idx, _, _ = train_test_split(
            train_idx, train_y, test_size=val_ratio_adj, stratify=train_y, random_state=random_state
        )

        return (torch.utils.data.Subset(dataset, train_idx),
                torch.utils.data.Subset(dataset, val_idx),
                torch.utils.data.Subset(dataset, test_idx))

    def train_epoch(self, loader):
        """
        Runs one single training epoch.

        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch in tqdm.tqdm(loader, desc="Training Epoch: "):
            batch = batch.to(self.device)
            labels = batch.y.view(-1).long()
            self.optimizer.zero_grad()
            logits = self.model(batch)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        return total_loss / total, correct / total

    def evaluate(self, loader):
        """
        Evaluates the model on a given data loader.

        Returns:
            tuple: (average_loss, accuracy, predictions, targets)
        """
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                labels = batch.y.view(-1).long()
                logits = self.model(batch)
                total_loss += self.criterion(logits, labels).item() * labels.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
        return total_loss / total, correct / total, np.array(all_preds), np.array(all_targets)

    def fit(self, train_loader, val_loader, epochs, save_path):
            """
            Executes the full training process with real-time logging.
            Saves the model state when Validation Accuracy improves.

            Args:
                train_loader (DataLoader): Loader for training data.
                val_loader (DataLoader): Loader for validation data.
                epochs (int): Number of training epochs.
                save_path (str): File path to save the best model weights.
            """
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            best_val_acc = 0.0
            
            pbar = tqdm.tqdm(range(1, epochs + 1))
            for epoch in pbar:
                # Execute one training epoch
                train_loss, train_acc = self.train_epoch(train_loader)
                
                # Execute validation
                val_loss, val_acc, _, _ = self.evaluate(val_loader)
                
                # Update learning rate scheduler based on Val Loss
                self.scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Save the best model logic
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(self.model.state_dict(), save_path)
                
                # Monitoring all metrics (Now including Val Loss)
                pbar.set_postfix({
                    f"EPOCH [{epochs}/{epoch}]"
                    'Train_Loss': f"{train_loss:.3f}",
                    'Train_Acc': f"{train_acc*100:.1f}%",
                    'Val_Loss': f"{val_loss:.3f}",  # <--- Added Val Loss here
                    'Val_Acc': f"{val_acc*100:.1f}%",
                    'LR': f"{current_lr:.1e}"
                })
                
            print(f"\n[INFO] Training complete. Best Val Acc: {best_val_acc*100:.2f}%")

def train(model, device, out_file="best_model.pth", epochs=50, lr=0.001, batch_size=32):
    """
    Main entry point for the training pipeline.
    
    This function handles dataset initialization, data splitting, training, 
    and generates a final performance report including a Confusion Matrix.
    """
    # 1. Dataset Initialization
    full_dataset = Image2Graph(
        root_dir=DATASET_ROOT, classes=CLASSES,
        grid_h=GRID_H, grid_w=GRID_W, image_h=IMAGE_H, image_w=IMAGE_W
    )
    
    # 2. Data Splitting
    train_ds, val_ds, test_ds = GNNTrainer.split_dataset(full_dataset)
    
    train_loader = torch_geometric.loader.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch_geometric.loader.DataLoader(val_ds, batch_size=batch_size)
    test_loader = torch_geometric.loader.DataLoader(test_ds, batch_size=batch_size)

    # 3. Training Execution
    trainer = GNNTrainer(model, device, lr=lr)
    trainer.fit(train_loader, val_loader, epochs, out_file)

    # 4. Final Evaluation on Test Set
    print("\n" + "="*50)
    print("FINAL TEST DATA PERFORMANCE REPORT")
    print("="*50)
    
    # Load the best saved model state
    model.load_state_dict(torch.load(out_file))
    test_loss, test_acc, preds, targets = trainer.evaluate(test_loader)
    
    # Metrics generation
    report = classification_report(targets, preds, target_names=CLASSES, zero_division=0)
    cm = confusion_matrix(targets, preds)
    
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test Loss:     {test_loss:.4f}")
    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", cm)
    
    # Save results to a text file
    res_path = out_file.replace(".pth", "_results.txt")
    with open(res_path, "w", encoding="utf-8") as f:
        f.write(f"Test Accuracy: {test_acc*100:.2f}%\n")
        f.write(f"Test Loss: {test_loss:.4f}\n\n")
        f.write("Classification Report:\n" + report + "\n\n")
        f.write("Confusion Matrix:\n" + np.array2string(cm))

    print(f"\n[SUCCESS] Final report saved to: {res_path}")