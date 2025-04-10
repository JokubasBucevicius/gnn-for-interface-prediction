import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pickle

from data_loader import DataLoader
from model import ProteinGAT
from loss_functions import focal_loss_with_logits


def train_one_epoch(model, data_loader, optimizer, criterion, device, threshold):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in data_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch)  # [n_nodes] for binary, [n_nodes, 3] for multiclass

        if model.mode == "binary":
            loss_per_node = criterion(out, batch.y.float())

            # Apply node weights to loss (element-wise multiplication)
            weighted_loss = loss_per_node.mean()

            preds = (out > threshold).long()
        else:
            loss_per_node = criterion(out, batch.y)

            # Apply node weights for multiclass too (optional but can help)
            weighted_loss = loss_per_node.mean()

            preds = out.argmax(dim=1)

        weighted_loss.backward()
        optimizer.step()

        total_loss += weighted_loss.item()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())

    avg_loss = total_loss / len(data_loader)

    if model.mode == "binary":
        acc = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        print(f"Train Loss: {avg_loss:.4f}, Train Accuracy: {acc:.4f}")
        print(f"Train Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    else:
        f1 = f1_score(all_labels, all_preds, average='weighted')
        print(f"Train Loss: {avg_loss:.4f}, Node F1 Score: {f1:.4f}")

    return avg_loss


def train_model(graphs, batch_size, mode, num_epochs, learning_rate, hidden_dim, heads, device, model_name, threshold, test_graphs):
    from torch_geometric.data import DataLoader as PyGDataLoader
    from model import ProteinGAT

    pyg_graphs = [g["pyg_graph"] for g in graphs]
    train_loader = PyGDataLoader(pyg_graphs, batch_size=batch_size, shuffle=True)

    
    output_dim = 1 if mode == "binary" else 3

    model = ProteinGAT(hidden_dim, output_dim, heads=heads, mode=mode).to(device) # add use_sas_weight=True if using node weights

    if mode == "binary":
        # Use BCEWithLogitsLoss (reduction='none') — allows us to apply per-node weights later
        criterion = lambda inputs, targets: focal_loss_with_logits(
            inputs, targets,
            alpha=0.745,
            gamma=2,
            reduction='none'
        )   # try out Focal Loss instead, with alpha parameter, add the class weights, additional information (Train accuracy and train f1)
    else:
        criterion = nn.CrossEntropyLoss(reduction='none')  # Multiclass also needs per-node weights support

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_one_epoch(model, train_loader, optimizer, criterion, device, threshold)

    # Save model
    model_path = f"trained_models/{model_name}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Save test dataset
    test_data_path = f"trained_models/{model_name}_test_graphs.pkl"
    with open(test_data_path, "wb") as f:
        pickle.dump(test_graphs, f)
    print(f"Test dataset saved to {test_data_path}")


