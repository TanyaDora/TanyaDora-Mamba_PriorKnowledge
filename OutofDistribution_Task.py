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

def create_data_loaders(args, input_features, target_features, train_data, test_data, train_samples, val_samples, test_samples):
    """
    Prepare train/val/test DataLoaders from raw CSV DataFrames.
    """
    train_dataset = TimeSeriesDataset(
        data=train_data,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        target_features=target_features,
        input_features=input_features,
        scale=True
    )

    test_dataset = TimeSeriesDataset(
        data=test_data,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        target_features=target_features,
        input_features=input_features,
        scale=True
    )

    # Fixed indices for train, validation, and test
    train_indices = np.arange(0, train_samples)
    val_indices = np.arange(train_samples, train_samples + val_samples)
    test_indices = np.arange(0, test_samples)

    # Subsets
    train_subset = SubsetWithTransform(train_dataset, train_indices)
    val_subset = SubsetWithTransform(train_dataset, val_indices)
    test_subset = SubsetWithTransform(test_dataset, test_indices)

    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader , train_indices, val_indices, test_indices

# Create experiment folders for saving models, plots, and results.
def create_experiment_folders(use_prior):
    base_dir = "ood_64_7" if use_prior else "ood_64_7_wp"
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
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

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
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in train_loop:
            optimizer.zero_grad()
            x, y, x_mark, y_mark = process_batch(batch, device)
            outputs = model(x, x_mark, y, y_mark)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            train_steps += 1
            train_loop.set_postfix(loss=total_train_loss / train_steps)
        avg_train_loss = total_train_loss / train_steps
        train_losses.append(avg_train_loss)

        avg_val_loss, _, _ = evaluate_model(model, val_loader, device, criterion)
        val_losses.append(avg_val_loss)
        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}")

        # Store the top N models based on validation loss
        if len(saved_models) < top_n:
            saved_models.append((avg_val_loss, model.state_dict()))
            saved_models.sort(key=lambda x: x[0])  # Sort by validation loss
        else:
            if avg_val_loss < saved_models[-1][0]:
                saved_models[-1] = (avg_val_loss, model.state_dict())
                saved_models.sort(key=lambda x: x[0])  # Sort by validation loss
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

# Reconstruction Function for Continuous Series
def reconstruct_continuous_series(predictions, seq_len, pred_len, test_indices):
    """
    Reconstructs a continuous series from overlapping windowed predictions.
    Uses averaging where windows overlap.
    """
    start_idx = test_indices[0] + seq_len
    end_idx = test_indices[-1] + seq_len + pred_len  
    total_length = end_idx - start_idx
    n_features = predictions.shape[2]

    # Prepare accumulators for predictions and counts.
    reconstructed = np.zeros((total_length, n_features))
    count = np.zeros((total_length, n_features))

    # Loop through each test sample
    for j, sample_idx in enumerate(test_indices):
        
        t_start = (sample_idx + seq_len) - start_idx
        t_end = t_start + pred_len
        t_end = min(t_end, total_length)
        # Sum the predictions into the appropriate segment.
        reconstructed[t_start:t_end, :] += predictions[j, :t_end - t_start, :]
        count[t_start:t_end, :] += 1

    # Average overlapping predictions (avoid division by zero)
    reconstructed = np.divide(reconstructed, count, out=reconstructed, where=count != 0)
    t_axis = np.arange(total_length)
    return reconstructed, t_axis

# -------------------------------
# Main Function.
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description='Mamba Time Series Forecasting')
    parser.add_argument('--train_data_path', type=str, required=True, help='Path to the training CSV data file')
    parser.add_argument('--test_data_path', type=str, required=True, help='Path to the testing CSV data file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size (only for validation split)')
    parser.add_argument('--val_size', type=float, default=0.1, help='Validation set size')
    parser.add_argument('--seq_len', type=int, default=64, help='Input sequence length')
    parser.add_argument('--pred_len', type=int, default=64, help='Prediction sequence length')
    parser.add_argument('--n_runs', type=int, default=5, help='Number of runs with different seeds')
    args = parser.parse_args()

    # Load training and testing data
    train_data = pd.read_csv(args.train_data_path)
    test_data = pd.read_csv(args.test_data_path)

    input_features = ['GPP', 'ET', 'PAR', 'Tair', 'VPD', 'Precip', 'fapar', 'CO2'] # Add these for integration of prior knowldge -  'GPP_pred', 'ET_pred'
    target_features = ['GPP', 'ET']

    # Number of samples for training, validation, and testing
    train_samples = int(len(train_data) * (1 - args.test_size - args.val_size))
    val_samples = int(len(train_data) * args.val_size)
    test_samples = int(len(test_data) * args.test_size)

    # -------------------------------
    # Dataloaders and Configs
    # -------------------------------
     # Create dataset and loaders for training/validation using the train_data and test_data
    train_loader, val_loader, test_loader, train_indices, val_indices, test_indices = create_data_loaders(
        args, input_features, target_features, train_data, test_data, train_samples, val_samples, test_samples
    )
    experiment_dirs = create_experiment_folders(use_prior=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    
    convergence_rates = {}
    early_stopping_epochs = {}
    all_saved_models = []
    experiment_classes = [Experiment1, Experiment2, Experiment3, Experiment4, Experiment5, Experiment6]

    for experiment_class in experiment_classes:
        if experiment_class.__name__ not in early_stopping_epochs:
            early_stopping_epochs[experiment_class.__name__] = []
        if experiment_class.__name__ not in convergence_rates:
            convergence_rates[experiment_class.__name__] = []
        print(f"\nRunning {experiment_class.__name__}")

        # Setup config for each experiment
        config = MambaConfig()
        config.seq_len = args.seq_len
        config.pred_len = args.pred_len
        config.enc_in = len(input_features)
        config.target_dim = len(target_features)
        config.input_features = input_features
        config.target_features = target_features

        # List to store predictions of the top 5 models
        all_predictions = []
        all_ground_truth = []
        all_train_losses = []
        all_val_losses = []

        # Loop to train multiple runs with different seeds
        for run in range(args.n_runs):
            print(f"\nRun {run+1}/{args.n_runs}")

            # Set random seed for each run
            seed = 5 + run  # Different seed for each run
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            # Init model and wrap it
            model = experiment_class(config)
            model = TargetSelector(model, input_features, target_features)

            # Train and save the best models for each run
            #model_save_path = os.path.join(experiment_dirs[experiment_class.__name__]['models'], f'top_model.pth')
            saved_models, train_losses, val_losses, lr_history, early_stop_epoch = train_model(
                model, train_loader, val_loader, device,
                epochs=args.epochs, lr=args.lr,
                patience=args.patience,experiment_class=experiment_class,  experiment_dirs=experiment_dirs,
                top_n=5  # Save top 5 models
            )

            all_train_losses.append(train_losses)
            all_val_losses.append(val_losses)
            early_stopping_epochs[experiment_class.__name__].append(early_stop_epoch)
            all_saved_models.extend(saved_models)
        all_saved_models.sort(key=lambda x: x[0])  # Sort by validation loss
        top_models = all_saved_models[:5]
        # Save the top 5 models after all runs
        for idx, (val_loss, state_dict) in enumerate(top_models):
            model_save_path = os.path.join(experiment_dirs[experiment_class.__name__]['models'], f'top_model_{idx+1}.pth')
            torch.save(state_dict, model_save_path)

        print("Top models saved after all runs.")

        # After all runs, evaluate the top models on the test data
        print("\nEvaluating top models on the test set and averaging predictions:")
        all_model_metrics = {}
        for i in range(1, 6):  # Loop over the top 5 models
            model_path = os.path.join(experiment_dirs[experiment_class.__name__]['models'], f'top_model_{i}.pth')  # Load each top model
            print(f"\nEvaluating {model_path}")
            model = experiment_class(config)
            model = TargetSelector(model, input_features, target_features)
            model.to(device)
            model.load_state_dict(torch.load(model_path))  # Load model state
            metrics, predictions, ground_truth = test_model(
                model, test_loader, device, target_features,  # Call the test_model function
                plot_save_path=os.path.join(experiment_dirs[experiment_class.__name__]['plots'], f'test_predictions_model_{i}.png')
            )
            model_name = f"top_model_{i}"
            all_model_metrics[model_name] = metrics

            # Store predictions and ground truth for averaging
            all_predictions.append(predictions)
            all_ground_truth.append(ground_truth)

        # Average predictions across the top 5 models
        averaged_predictions = np.mean(all_predictions, axis=0)
        ground_truth = np.concatenate(all_ground_truth, axis=0)  # Concatenate ground truth from all top models
        # Calculate convergence rate and early stopping epoch
        for run_idx in range(args.n_runs):
            val_loss = np.array(all_val_losses[run_idx])
            delta_loss = np.diff(val_loss)
            avg_convergence_rate = np.mean(delta_loss[delta_loss < 0])
            convergence_rates[experiment_class.__name__].append(avg_convergence_rate)

        # Save results to a text file
        result_file_path = os.path.join(experiment_dirs[experiment_class.__name__]['results'], 'convergence_and_early_stopping.txt')
        with open(result_file_path, 'w') as f:
            f.write("Convergence Rate and Early Stopping Epochs for Each Experiment:\n")
            f.write("\nConvergence Rates:\n")
            for exp_name, rate in convergence_rates.items():
                f.write(f"{exp_name}: {np.mean(rate):.6f}\n")

            f.write("\nEarly Stopping Epochs:\n")
            for exp_name, epochs in early_stopping_epochs.items():
                f.write(f"{exp_name}: {np.mean(epochs):.2f}\n")

        print(f"Results saved to {result_file_path}\n")
        pr_values = []
        with torch.no_grad():
            for batch in test_loader:
                # Extract input features (including GPP_pred and ET_pred)
                x, y, x_mark, y_mark = process_batch(batch, device)

                # Extract GPP_pred and ET_pred from the input features
                pr_gpp = x[:, :, input_features.index('GPP_pred')].cpu().numpy()
                pr_et = x[:, :, input_features.index('ET_pred')].cpu().numpy()

                # Combine GPP_pred and ET_pred into a single 2-column array
                combined = np.stack((pr_gpp, pr_et), axis=-1)  # This stacks them along the last axis (columns)

                # Append to the list
                pr_values.append(combined)
        max_epochs = max(len(run) for run in all_train_losses)  # Find the longest sequence

        # Pad each run's loss to the max_epochs length with np.nan
        padded_train_losses = [np.pad(run, (0, max_epochs - len(run)), constant_values=np.nan) for run in all_train_losses]
        padded_val_losses = [np.pad(run, (0, max_epochs - len(run)), constant_values=np.nan) for run in all_val_losses]

        # Now calculate the mean of the padded losses, ignoring NaNs
        avg_train_loss = np.nanmean(padded_train_losses, axis=0)
        avg_val_loss = np.nanmean(padded_val_losses, axis=0)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(avg_train_loss) + 1), avg_train_loss, label='Average Train Loss', color='blue')
        plt.plot(range(1, len(avg_val_loss) + 1), avg_val_loss, label='Average Validation Loss', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Average Train and Validation Loss over Epochs')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(experiment_dirs[experiment_class.__name__]['plots'], 'average_train_val_loss.png'), dpi=300)
        plt.close()

        # Plot Best Model Loss (for the top model)
        best_train_loss = min(all_train_losses, key=lambda x: np.mean(x))  # Best training loss across runs
        best_val_loss = min(all_val_losses, key=lambda x: np.mean(x))      # Best validation loss across runs

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(best_train_loss) + 1), best_train_loss, label='Best Train Loss', color='blue')
        plt.plot(range(1, len(best_val_loss) + 1), best_val_loss, label='Best Validation Loss', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Best Train and Validation Loss over Epochs')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(experiment_dirs[experiment_class.__name__]['plots'], 'best_train_val_loss.png'), dpi=300)
        plt.close()

        # Concatenate across all batches
        #pr_values_combined = np.concatenate(pr_values, axis=0)
        reconstructed_pred, t_axis = reconstruct_continuous_series(averaged_predictions, args.seq_len, args.pred_len, test_indices)
        reconstructed_true, _ = reconstruct_continuous_series(ground_truth, args.seq_len, args.pred_len, test_indices)
        reconstructed_pr, _ = reconstruct_continuous_series(pr_values_combined, args.seq_len, args.pred_len, test_indices)
        # Plot for GPP (Averaged predictions)
        plt.figure(figsize=(15, 6))
        plt.plot(t_axis[:400], reconstructed_true[:400, 0], label='Ground Truth GPP')
        plt.plot(t_axis[:400], reconstructed_pred[:400, 0], label='Predicted GPP', color='red', linewidth=2)
        plt.plot(t_axis[:400], reconstructed_pr[:400, 0], label='Prior Knowldge GPP', color='green', linewidth=2)
        plt.xlabel('Time Step', fontsize=16)
        plt.ylabel('GPP Value', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.gca().set_ylim(top=1)
        plt.savefig(os.path.join(experiment_dirs[experiment_class.__name__]['plots'], 'prediction_GPP.png'), dpi=300)
        plt.close()

        # Plot for ET (Averaged predictions)
        plt.figure(figsize=(15, 6))
        plt.plot(t_axis[:400], reconstructed_true[:400, 1], label='Ground Truth ET')
        plt.plot(t_axis[:400], reconstructed_pred[:400, 1], label='Predicted ET', color='red', linewidth=2)
        plt.plot(t_axis[:400], reconstructed_pr[:400, 1], label='Prior Knowldge ET', color='green', linewidth=2)
        plt.xlabel('Time Step', fontsize=16)
        plt.ylabel('ET Value', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.gca().set_ylim(top=1)
        plt.savefig(os.path.join(experiment_dirs[experiment_class.__name__]['plots'], 'prediction_ET.png'), dpi=300)
        plt.close()


        # Save losses and metrics for the averaged predictions
        result_file = os.path.join(experiment_dirs[experiment_class.__name__]['results'], 'losses_all_top_models.txt')
        with open(result_file, 'w') as f:
            f.write("Test Metrics per Feature for Each Top Model:\n")
            for model_name, model_metrics in all_model_metrics.items():
                f.write(f"\n{model_name}:\n")
                for feature, metric_data in model_metrics.items():
                    f.write(f"\n  {feature}:\n")
                    for metric_name, value in metric_data.items():
                        f.write(f"    {metric_name}: {value:.6f}\n")
            f.write("\n\nAverage Metrics Across Top 5 Models:\n")
            average_metrics = {}

            for feature in target_features:
                metric_sums = {}
                metric_counts = {}

                for model_metrics in all_model_metrics.values():
                    feature_metrics = model_metrics[feature]
                    for metric_name, value in feature_metrics.items():
                        metric_sums[metric_name] = metric_sums.get(metric_name, 0.0) + value
                        metric_counts[metric_name] = metric_counts.get(metric_name, 0) + 1

                average_metrics[feature] = {
                    metric_name: metric_sums[metric_name] / metric_counts[metric_name]
                    for metric_name in metric_sums
                }

            for feature, metric_data in average_metrics.items():
                f.write(f"\n  {feature}:\n")
                for metric_name, value in metric_data.items():
                    f.write(f"    {metric_name}: {value:.6f}\n")

    print("All experiments completed!")

if __name__ == "__main__":
    main()