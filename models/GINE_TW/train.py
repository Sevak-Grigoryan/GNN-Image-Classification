import os
import tqdm
import torch
import psutil
import numpy as np
import torch_geometric
from directions import *       
from model import GINE
from data_processing import Image2Graph
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix



print(DEVICE)


def print_memory_usage():
    if DEVICE == "cuda":
        print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB / "
              f"{torch.cuda.max_memory_allocated()/1e9:.2f}GB (max)")
        torch.cuda.reset_peak_memory_stats()
    
    process = psutil.Process()
    print(f"RAM Usage: {process.memory_info().rss/1e9:.2f}GB")

def stratified_train_val_test_split(
        dataset: torch_geometric.data.Dataset,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42
):

    indices = np.arange(len(dataset))
    labels = [label for _, label in dataset.samples]

    # First: train+val vs test
    train_indices, test_indices, trainval_y, test_y = train_test_split(
        indices,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )

    val_ratio_within_trainval = val_size / (1.0 - test_size)

    train_idx, val_idx, _, _ = train_test_split(
        train_indices,
        trainval_y,
        test_size=val_ratio_within_trainval,
        stratify=trainval_y,
        random_state=random_state,
    )

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset   = torch.utils.data.Subset(dataset, val_idx)
    test_dataset  = torch.utils.data.Subset(dataset, test_indices)

    print(
        f"[SPLIT] Train: {len(train_dataset)} | "
        f"Val: {len(val_dataset)} | "
        f"Test: {len(test_dataset)}"
    )
    return train_dataset, val_dataset, test_dataset

def train_one_epoch(
        model: torch.nn.Module,
        data_loader: torch_geometric.loader.DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer
):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm.tqdm(data_loader, desc="Training Epochs"):
        batch = batch.cuda()
        labels = batch.y.view(-1).long()

        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    print_memory_usage()    
    # --- Compute gradient norm after the epoch ---
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    print(f"Epoch Gradient Norm: {total_norm:.4f}")

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy

def train_one_epoch(
        model: torch.nn.Module,
        data_loader: torch_geometric.loader.DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer
):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm.tqdm(data_loader, desc="Training Epochs"):
        batch = batch.cuda()

        # labels: (num_graphs,) long
        labels = batch.y.view(-1).long()

        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm.tqdm(loader, desc="Training Epochs"):
            batch = batch.cuda()

            logits = model(batch)
            labels = batch.y.view(-1).long()

            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc


def collect_predictions(model, loader):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.cuda()
            logits = model(batch)
            preds = logits.argmax(dim=1)
            targets = batch.y.view(-1).long()

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    return np.array(all_preds), np.array(all_targets)


def main():
    full_dataset = Image2Graph(
        root_dir=DATASET_ROOT,
        classes=CLASSES,
        grid_h=GRID_H,
        grid_w=GRID_W,

    )

    all_labels = [label for _, label in full_dataset.samples]
    print("Unique labels in dataset:", sorted(set(all_labels)))
    print(f"[INFO] Total samples: {len(full_dataset)}")

    train_dataset, val_dataset, test_dataset = stratified_train_val_test_split(
        full_dataset,
        val_size=0.15,
        test_size=0.15,
        random_state=42,
    )

    train_loader = torch_geometric.loader.DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_loader = torch_geometric.loader.DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False
    )
    test_loader = torch_geometric.loader.DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False
    )

    model = GINE().cuda()
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4,
    )

    # weights = torch.tensor([2.0, 3.0, 1.0], dtype=torch.float, device=DEVICE)
    criterion = torch.nn.CrossEntropyLoss()

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=10,     # шаги до первого рестарта
    #     T_mult=2,   # увеличиваем период рестарта каждый раз
    #     eta_min=1e-6
    # )
    # scaler = torch.cuda.amp.GradScaler(enabled=True) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=6,
        min_lr=1e-6
    )

    print(
        f"[INFO] Train: {len(train_dataset)} | "
        f"Val: {len(val_dataset)} | "
        f"Test: {len(test_dataset)}"
    )

    best_val_acc = 0.0
    best_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GINE_TW_2.0_ERIK_HAYKO_100.pth")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print("\n"
            f"Epoch [{epoch}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% "
            f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% "
            f"| LR: {current_lr:.6f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved (val_acc = {best_val_acc*100:.2f}%)")

    print("\n[INFO] Training finished.")
    print(f"[INFO] Best validation accuracy: {best_val_acc*100:.2f}%")

    if os.path.exists(best_model_path):
        print("\n[INFO] Loading best model for TEST evaluation...")
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    else:
        print(f"[WARN] Best model file '{best_model_path}' not found. Using last epoch model.")

    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"\nTEST Accuracy: {test_acc*100:.2f}%")
    print(f"TEST Loss: {test_loss:.4f}")

    preds, targets = collect_predictions(model, test_loader)

    print("\n=== CLASSIFICATION REPORT (TEST) ===")
    labels = list(range(len(CLASSES)))
    report = classification_report(
        targets,
        preds,
        labels=labels,
        target_names=CLASSES,
        zero_division=0,
    )
    print(report)

    print("=== CONFUSION MATRIX (TEST) ===")
    cm = confusion_matrix(targets, preds, labels=labels)
    print(cm)

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "GINE_TW_2.0_ERIK_HAYKO_100.txt"), "w", encoding="utf-8") as f:
        f.write("Classification Report (TEST)\n")
        f.write(report)
        f.write("\nConfusion Matrix\n")
        for row in cm:
            f.write(" ".join(str(x) for x in row) + "\n")


if __name__ == "__main__":
    main()

