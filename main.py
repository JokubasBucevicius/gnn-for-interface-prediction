"""
Main script for training and evaluating the Graph Neural Network.
"""

import torch
import numpy as np
import os
from data_loader import DataLoader
from train import train_model
from evaluate import evaluate_model, list_saved_models

def main():
    # Configuration variables
    base_path = "../grafai/graphs/be_klasterio/dgDNR/"
    batch_size = 6
    mode = "binary"
    num_epochs = 250
    learning_rate = 0.0018
    hidden_dim = 128
    heads = 4
    threshold = 0.3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print(" CUDA is available!")
        print("Device:", torch.cuda.get_device_name(0))
        print("Memory total:", torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")
    else:
        print("CUDA not available")

    # Ask user whether to train or evaluate
    action = input("Choose action: [train/evaluate] ").strip().lower()

    if action == "train":
        # Load dataset only for training
        loader = DataLoader(base_path, batch_size, mode)
        train_graphs, test_graphs = loader.load_and_split(train_ratio=0.8)

        # Check label and surface imbalance before training
        # print("\nChecking label and surface imbalance before training...")
        # all_labels = np.concatenate([g["pyg_graph"].y.cpu().numpy() for g in train_graphs])
        # all_surface = np.concatenate([g["nodes"]["surface_atom"].values for g in train_graphs])

        # num_binding = np.sum(all_labels)
        # num_surface = np.sum(all_surface)
        # num_total = len(all_labels)

        # binding_percentage = (num_binding / num_total) * 100
        # surface_percentage = (num_surface / num_total) * 100
        # surface_binding_percentage = (num_binding / num_surface) * 100
        # binding_probs = loader.calculate_binding_probabilities(train_graphs)

        # print(f"Total Training Nodes: {num_total}")
        # print(f"Surface Nodes: {num_surface} ({surface_percentage:.2f}%)")
        # print(f"Binding Nodes: {num_binding} ({binding_percentage:.2f}%)")
        # print(f"Surface Nodes/Binding Nodes: {surface_binding_percentage:.2f}%")
        # print("Residue binding probabilities (calculated from training data):")
        # for res_type, prob in binding_probs.items():
        #     print(f"Residue {res_type}: {prob:.3f}")

        # print("-" * 40)

        # Ask for model name and train
        model_name = input("Enter model name (e.g., GAT_dgDNR_binary_lr00001_hidden64): ").strip()
        train_model(train_graphs, batch_size, mode, num_epochs, learning_rate, hidden_dim, heads, device, model_name, threshold, test_graphs)

    elif action == "evaluate":
        # List saved models
        models = list_saved_models()
        if not models:
            print("No saved models to evaluate. Please train a model first.")
            return

        choice = int(input("Select model (enter number): ").strip())
        if choice < 1 or choice > len(models):
            print("Invalid selection.")
            return

        model_name = models[choice - 1]
        model_path = f"trained_models/{model_name}"

        # Evaluate without loading full dataset (using saved test graphs)
        evaluate_model(batch_size, mode, model_path, device)

    else:
        print("Invalid action. Please choose 'train' or 'evaluate'.")

if __name__ == "__main__":
    main()

