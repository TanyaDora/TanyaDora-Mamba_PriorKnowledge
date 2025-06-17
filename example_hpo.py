# Import necessary libraries
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import argparse
import sys
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import mean_absolute_percentage_error, r2_score

# Import experiments from your SMamba_Experiment file
smamba_path = 'path for SMamba_Experiment.py'  # Update this path to your actual file location
sys.path.append(os.path.dirname(smamba_path))
from SMamba_Experiment import Experiment1, Experiment2, Experiment3, Experiment4, Experiment5, Experiment6


# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Dataset and Helper Classes
class TimeSeriesDataset(Dataset):
    """
    This class is used to preprocess time-series data and prepare it for training a forecasting model.
    The data is split into input-output pairs,features are scaled using MinMaxScaler.
    """
    def __init__(self, data, seq_len, pred_len, target_features, input_features, scale=True):
        super(TimeSeriesDataset, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_features = target_features
        self.input_features = input_features
        self.scale = scale

        self.scalers = {}
        self.data = data.copy()
        if self.scale:
            for feature in self.input_features:
                self.scalers[feature] = MinMaxScaler()
                self.data[feature] = self.scalers[feature].fit_transform(
                    self.data[feature].values.reshape(-1, 1)
                )
            for feature in self.target_features:
                self.scalers[feature] = MinMaxScaler()
                self.data[feature] = self.scalers[feature].fit_transform(
                    self.data[feature].values.reshape(-1, 1)
                )

        self.samples = self._create_samples()

    def _create_samples(self):
        """
        Creates input-output pairs from the time-series data based on seq_len and pred_len.
        Each sample consists of a sequence of input features and the corresponding target values.
        """
        samples = []
        total_samples = len(self.data) - self.seq_len - self.pred_len + 1
        for i in range(total_samples):
            x_start = i
            x_end = i + self.seq_len
            y_start = x_end
            y_end = y_start + self.pred_len

            # Input data for all input features
            x_data = self.data[self.input_features].iloc[x_start:x_end].values
            # Target data using only target features
            y_data = self.data[self.target_features].iloc[y_start:y_end].values

            x_mark = np.zeros((self.seq_len, len(self.input_features)))
            y_mark = np.zeros((self.pred_len, len(self.input_features)))

            samples.append({
                'x': x_data,
                'y': y_data,
                'x_mark': x_mark,
                'y_mark': y_mark
            })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'x': torch.FloatTensor(sample['x']),
            'y': torch.FloatTensor(sample['y']),
            'x_mark': torch.FloatTensor(sample['x_mark']),
            'y_mark': torch.FloatTensor(sample['y_mark'])
        }

# A helper dataset class to select subsets by indices.
class SubsetWithTransform(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

# Model Configuration and Wrapper
class MambaConfig:
    def __init__(self):
        # Model parameters
        self.num_heads = 16
        self.seq_len = 0  # to be set later
        self.pred_len = 0  # to be set later
        self.d_model = 512
        self.d_state = 16
        self.d_ff = 256
        self.e_layers = 5
        self.dropout = 0.1
        self.activation = 'gelu'
        self.embed = 'fixed'
        self.freq = 'h'
        self.output_attention = False
        self.use_norm = False
        self.class_strategy = None

# Wraps model and selects only target features from output.
class TargetSelector(nn.Module):
    def __init__(self, model, input_features, target_features):
        super(TargetSelector, self).__init__()
        self.model = model
        self.input_features = input_features
        self.target_features = target_features
        self.target_indices = [input_features.index(feature) for feature in target_features]

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        outputs = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=mask)
        selected_outputs = outputs[:, :, self.target_indices]
        return selected_outputs

# -------------------------------
# Helper Functions
# -------------------------------

def process_batch(batch, device):
    """
    Move batch tensors to the specified device (CPU or CUDA).
    """
    x = batch['x'].to(device)
    y = batch['y'].to(device)
    x_mark = batch['x_mark'].to(device)
    y_mark = batch['y_mark'].to(device)
    return x, y, x_mark, y_mark

def evaluate_model(model, loader, device, criterion=nn.MSELoss()):
    """
    Evaluate model over a dataloader and return average loss + all predictions.
    """
    model.eval()
    total_loss = 0
    total_steps = 0
    predictions = []
    ground_truth = []
    with torch.no_grad():
        for batch in loader:
            x, y, x_mark, y_mark = process_batch(batch, device)
            outputs = model(x, x_mark, y, y_mark)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            total_steps += 1
            predictions.append(outputs.cpu().numpy())
            ground_truth.append(y.cpu().numpy())
    avg_loss = total_loss / total_steps
    predictions = np.concatenate(predictions, axis=0)
    ground_truth = np.concatenate(ground_truth, axis=0)
    return avg_loss, predictions, ground_truth

# Create DataLoaders using a temporal split.
def create_data_loaders(args, input_features, target_features, data):
    total_samples = len(data) - args.seq_len - args.pred_len + 1
    if total_samples <= 0:
        raise ValueError("Dataset is too small for the specified seq_len and pred_len.")

    # Temporal split indices for training, validation, and test sets.
    test_size = int(total_samples * args.test_size)
    train_val_size = total_samples - test_size
    val_size = int(train_val_size * args.val_size)
    train_size = train_val_size - val_size

    train_indices = np.arange(0, train_size)
    val_indices = np.arange(train_size, train_size + val_size)
    test_indices = np.arange(train_val_size, total_samples)
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    print(f"Test samples: {test_size}")

    # Create dataset from the loaded DataFrame.
    dataset = TimeSeriesDataset(
        data=data,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        target_features=target_features,
        input_features=input_features,
        scale=True
    )

    train_loader = DataLoader(SubsetWithTransform(dataset, train_indices), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(SubsetWithTransform(dataset, val_indices), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(SubsetWithTransform(dataset, test_indices), batch_size=args.batch_size, shuffle=False)

    # Also return test_indices in case we need them for reconstruction.
    return dataset, train_loader, val_loader, test_loader, test_indices

# Create experiment folders for saving models, plots, and results.
def create_experiment_folders(use_prior):
    base_dir = "best_hyytiala_64_7_exp5" if use_prior else "best_12_64_30_wp"
    exp_dirs = {}
    for exp in ["Experiment1", "Experiment2", "Experiment3", "Experiment4", "Experiment5", "Experiment6"]:
        exp_dir = os.path.join(base_dir, exp)
        exp_dirs[exp] = {
            'models': os.path.join(exp_dir, 'models'),
            'plots': os.path.join(exp_dir, 'plots'),
            'results': os.path.join(exp_dir, 'results')
        }
        for subdir in exp_dirs[exp].values():
            os.makedirs(subdir, exist_ok=True)
    return exp_dirs

# Training function using the evaluation helper.
def train_model(model, train_loader, val_loader, device, epochs=100, lr=1e-4, patience=10,experiment_class=None, experiment_dirs=None, top_n=5):
    """
    Trains the model using the provided training and validation data.
    Implements early stopping based on validation loss and saves the top N models with the best validation performance.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss() # Loss function (Mean Squared Error)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6) # Learning rate scheduler

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    lr_history = []

    saved_models = [] # Store the top models based on validation loss
    early_stop_epoch = epochs # To store the epoch where early stopping occurred

    for epoch in range(epochs):
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)
        model.train()
        total_train_loss = 0
        train_steps = 0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")  # Training loop with progress bar
        for batch in train_loop:
            optimizer.zero_grad()
            x, y, x_mark, y_mark = process_batch(batch, device)
            outputs = model(x, x_mark, y, y_mark)
            loss = criterion(outputs, y)
            loss.backward() # Backpropagation
            optimizer.step() # Update weights
            total_train_loss += loss.item() 
            train_steps += 1
            train_loop.set_postfix(loss=total_train_loss / train_steps)
        avg_train_loss = total_train_loss / train_steps
        train_losses.append(avg_train_loss)
        # Evaluate on the validation set
        avg_val_loss, _, _ = evaluate_model(model, val_loader, device, criterion)
        val_losses.append(avg_val_loss)
        scheduler.step() # Update the learning rate
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}")

        # Store the top N models based on validation loss
        if len(saved_models) < top_n:
            saved_models.append((avg_val_loss, model.state_dict()))
            saved_models.sort(key=lambda x: x[0])  # Sort by validation loss
        else:
            if avg_val_loss < saved_models[-1][0]:
                saved_models[-1] = (avg_val_loss, model.state_dict())
                saved_models.sort(key=lambda x: x[0]) 
        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0  # Reset patience counter if validation loss improves
        else:
            patience_counter += 1
            print(f"Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss.")
                early_stop_epoch = epoch + 1
                break 

    return saved_models, train_losses, val_losses, lr_history, early_stop_epoch


def test_model(model, test_loader, device, target_features, plot_save_path='predictions.png'):
    """
    This function evaluates the trained model on the test data.
    It calculates various evaluation metrics (MSE, MAE, RMSE, MAPE, R2, and Tolerance Accuracy) 
    for each target feature and returns the computed metrics along with the predictions and ground truth.
    """
    _, predictions, ground_truth = evaluate_model(model, test_loader, device)
    metrics_per_feature = {}
    for i, feature in enumerate(target_features):
        pred = predictions[:, :, i]
        true = ground_truth[:, :, i]
        errors = (pred - true)
        squared_errors = errors ** 2
        mse = np.mean(squared_errors)
        mse_std = np.std(squared_errors)
        mae = np.mean(np.abs(pred - true))
        rmse = np.sqrt(mse)
        metrics_per_feature[feature] = {
            'MSE': mse,
            'MSE_STD': mse_std,
            'MAE': mae,
            'RMSE': rmse,
        }
        print(f"\nMetrics for {feature}:")
        print(f"MSE: {mse:.4f}")
        print(f"MSE: {mse:.4f} ± {mse_std:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
    return metrics_per_feature, predictions, ground_truth

# -------------------------------
# Hyperparameter Optimization (HPO) with Optuna
# -------------------------------

def objective(trial):
    """  Objective function for Optuna hyperparameter optimization.
    This function defines the hyperparameters to be optimized and trains a model with those parameters.
    It returns the validation loss as the objective value.
    """
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3) 
    dropout = trial.suggest_uniform('dropout', 0.1, 0.5)  
    seq_len = trial.suggest_categorical('seq_len', [32, 64, 128])

    # Fixed values for other hyperparameters
    batch_size = 32  
    pred_len = 7    
    d_model = 512   
    e_layers = 5     
    num_heads = 4
    # Re-create data loaders with the new sequence lengths and batch size.
    _, trial_train_loader, trial_val_loader, _ = create_data_loaders(args, input_features, target_features, data)

    # Create model configuration with updated hyperparameters.
    config = MambaConfig()
    config.seq_len = seq_len
    config.pred_len = pred_len
    config.enc_in = len(input_features)
    config.target_dim = len(target_features)
    config.input_features = input_features
    config.target_features = target_features
    config.d_model = d_model  # Fixed d_model
    config.dropout = dropout  # Fixed dropout
    config.e_layers = e_layers  # Fixed e_layers
    config.num_heads = num_heads  # Set num_heads from HPO

    # Choose an experiment (e.g., Experiment5)
    model = Experiment5(config)  # Experiment5 can be changed as needed
    model = TargetSelector(model, input_features, target_features)
    model.to(device)

    # Train the model for a reduced number of epochs for HPO.
    temp_model_save = "temp_best_model.pth"
    train_losses, val_losses, _ = train_model_func(
        model=model,
        train_loader=trial_train_loader,
        val_loader=trial_val_loader,
        device=device,
        epochs=10,
        lr=lr,
        patience=3,
        model_save_path=temp_model_save
    )
    
    # Return the final validation loss as the objective value.
    return val_losses[-1]

# -------------------------------
# Main Function
# -------------------------------

def main():
    global args, data, input_features, target_features, train_loader, val_loader, device
    parser = argparse.ArgumentParser(description='Mamba Time Series Forecasting with HPO')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV data file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--val_size', type=float, default=0.1, help='Validation set size')
    parser.add_argument('--seq_len', type=int, default=64, help='Input sequence length')
    parser.add_argument('--pred_len', type=int, default=16, help='Prediction sequence length')
    parser.add_argument('--hpo', action='store_true', help='Perform hyperparameter optimization')
    args = parser.parse_args()

    # Load CSV data once.
    data = pd.read_csv(args.data_path)

    experiment_dirs = create_experiment_folders(use_prior=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    input_features = ['GPP', 'ET', 'PAR', 'Tair', 'VPD', 'Precip', 'fapar', 'CO2', 'GPP_pred', 'ET_pred']
    target_features = ['GPP', 'ET']

    # Create data loaders.
    _, train_loader, val_loader, test_loader = create_data_loaders(args, input_features, target_features, data)

    if args.hpo:
        print("Starting Hyperparameter Optimization...")
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=25)
        print("Best hyperparameters:", study.best_trial.params)
        best_params = study.best_trial.params
        final_lr = best_params['lr']
        final_dropout = best_params['dropout']
        final_d_model = best_params['d_model']
        final_e_layers = best_params['e_layers']
        final_batch_size = best_params['batch_size']
        # Update args.batch_size with the best found value.
        args.batch_size = final_batch_size
    else:
        final_lr = args.lr
        final_dropout = MambaConfig().dropout
        final_d_model = MambaConfig().d_model
        final_e_layers = MambaConfig().e_layers

    # Re-create final data loaders with final batch size.
    _, train_loader, val_loader, test_loader = create_data_loaders(args, input_features, target_features, data)

    # Setup final configuration with chosen hyperparameters.
    config = MambaConfig()
    config.seq_len = args.seq_len
    config.pred_len = args.pred_len
    config.enc_in = len(input_features)
    config.target_dim = len(target_features)
    config.input_features = input_features
    config.target_features = target_features
    config.dropout = final_dropout
    config.d_model = final_d_model
    config.e_layers = final_e_layers

    # Choose experiment (e.g., Experiment4)
    experiment_classes = [Experiment1, Experiment2, Experiment3, Experiment4, Experiment5, Experiment6]
    for experiment_class in experiment_classes:
        print(f"\nRunning {experiment_class.__name__} with final hyperparameters")
        model = experiment_class(config)
        model = TargetSelector(model, input_features, target_features)

        # Train the final model.
        model_save_path = os.path.join(experiment_dirs[experiment_class.__name__]['models'], 'best_model.pth')
        train_losses, val_losses, lr_history = train_model_func(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            lr=final_lr,
            patience=args.patience,
            model_save_path=model_save_path
        )

        # Load best model and test.
        model.load_state_dict(torch.load(model_save_path))
        metrics, predictions, ground_truth = test_model_func(
            model=model,
            test_loader=test_loader,
            device=device,
            target_features=target_features,
            plot_save_path=os.path.join(experiment_dirs[experiment_class.__name__]['plots'], 'predictions.png')
        )

        # Save training curves, test metrics, and (if HPO was performed) best hyperparameters to losses.txt.
        losses_file_path = os.path.join(experiment_dirs[experiment_class.__name__]['results'], 'losses.txt')
        with open(losses_file_path, 'w') as f:
            if args.hpo:
                f.write("Best Hyperparameters from HPO:\n")
                for key, value in best_params.items():
                    f.write(f"{key}: {value}\n")
                f.write(f"Best Validation Loss during HPO: {study.best_trial.value}\n\n")
            f.write("Epoch\tTrain Loss\tValidation Loss\tLearning Rate\n")
            for epoch, (train_loss, val_loss, lr_val) in enumerate(zip(train_losses, val_losses, lr_history), 1):
                f.write(f"{epoch}\t{train_loss:.6f}\t{val_loss:.6f}\t{lr_val:.6f}\n")
            f.write("\nTest Metrics per Feature:\n")
            for feature, metrics_data in metrics.items():
                f.write(f"\n{feature}:\n")
                for metric_name, value in metrics_data.items():
                    f.write(f"{metric_name}: {value:.6f}\n")

        print("\nExperiment completed!")

if __name__ == "__main__":
    main()
