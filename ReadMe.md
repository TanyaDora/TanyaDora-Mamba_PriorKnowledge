# Evaluating Mamba Model Performance on Time-Series Datasets with Prior Knowledge Integration

##  Overview

This project investigates the use of the **Mamba** model — a Selective State Space Model — for **environmental time-series forecasting**, particularly in **boreal forest ecosystems**. The study explores how integrating **prior knowledge** from the **PRELES model** affects Mamba’s ability to generalize, capture long-term dependencies, and improve prediction accuracy.

##  Key Contributions

- Adaptation of the S-Mamba model for time-series forecasting using boreal forest datasets.
- Integration of ecological prior knowledge (GPP, ET from PRELES) into Mamba using:
  - Direct feature concatenation
  - Multi-head attention over all features
  - Selective attention over prior knowledge only
- Evaluation of model performance in **In-Distribution (ID)** and **Out-of-Distribution (OOD)** scenarios.

##  Methodology

- **Dataset**: Daily environmental variables (e.g., PAR, Tair, CO₂) from forest sites in Denmark and Finland.
- **Target Variables**: 
  - Gross Primary Production (GPP)
  - Evapotranspiration (ET)
- **Prior Knowledge**: Outputs from the PRELES model (GPP_pred, ET_pred).
- **Experiments**: Six variations of the model integrating prior knowledge in different ways (direct input, attention-based).
- **Metrics**: MAE, RMSE, MSE (± STD)

##  Experimental Setup

| Experiment | Integration Method                       |
|------------|-------------------------------------------|
| Exp 1      | Feature embedding + prior knowledge       |
| Exp 2      | Linear projection + prior knowledge       |
| Exp 3      | Multi-head attention after embedding      |
| Exp 4      | Multi-head attention before embedding     |
| Exp 5      | Selective attention on prior knowledge    |
| Exp 6      | Separate projection and attention fusion  |

### Environments

- **Optimizer**: AdamW
- **Scheduler**: Cosine annealing
- **Loss**: Mean Squared Error (MSE)
- **HPO**: Optuna (search over heads, dropout, batch size, etc.)


## Files and Structure

- `Indistribution_Task.py` – Run In-Distribution experiments
- `OutofDistribution_Task.py` – Run Out-of-Distribution experiments
- `SMamba_Experiment.py` – Experiment Logics
- `attention_layer.py` – Attention mechanisms 
- `requirements.txt` – Python dependencies
- `README.md`
---

## How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run In-Distribution experiments:

```bash
python Indistribution_Task.py --data_path=path to data.csv 
```

3. Run Out-of-Distribution experiments:

```bash
python OutofDistribution_Task.py.py --train_data_path= path to Train.csv --test_data_path=path to Test.csv 
```

## Notes

- To enable this, use the flag `--use_prior_knowledge True`. When this flag is set to true, the input features should include `GPP_pred` and `ET_pred`, which are prior predictions derived from the PRELES model. In this case, make sure that your `input_features` list in the code includes these two additional variables along with the standard ones like `GPP`, `ET`, `PAR`, `Tair`, `VPD`, `Precip`, `fapar`, and `CO2`.

- If prior knowledge is enabled, you should also uncomment the plotting lines in the code that visualize prior knowledge alongside the model's predictions. Specifically, these lines plot `reconstructed_pr[:400, 0]` and `reconstructed_pr[:400, 1]` to show prior-based GPP and ET values, respectively. These plots will be saved in the `plots/` directory along with others generated during training.

- If you are not using prior knowledge (`--use_prior_knowledge False`, the default), make sure to exclude `GPP_pred` and `ET_pred` from your `input_features`.

- After training, the model will save the top 5 checkpoints based on validation loss, averaged prediction plots for GPP and ET, evaluation metrics such as MSE, RMSE, and MAE, and convergence details in the `results/` and `plots/` folders.
