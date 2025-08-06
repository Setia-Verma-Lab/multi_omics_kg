import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.nn import HeteroConv, global_mean_pool, SAGEConv
from torch_geometric.loader import DataLoader
from data.patient_graph_dataset import PatientGraphDataset
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score, average_precision_score)
import numpy as np
import pandas as pd

def bootstrap_metric(y_true, y_score, metric_fn, n_boot=200, seed=0, is_prob=True):
    """
    Compute metric_fn over n_boot bootstrap resamples.
    If is_prob=False, metric_fn should take y_true, y_pred (hard).
    """
    rng = np.random.default_rng(seed)
    n   = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        ys = y_score[idx]
        if is_prob:
            vals.append(metric_fn(yt, ys))
        else:
            # for f1/accuracy: ys is hard pred
            vals.append(metric_fn(yt, ys.astype(int)))
    vals = np.array(vals)
    return float(np.mean(vals)), float(vals.std(ddof=1))

class HeteroGraphClassifier(pl.LightningModule):
    def __init__(self, num_layers, covariate_dim, hidden_dim=64):
        super().__init__()
        self.init_proj = torch.nn.Linear(1, hidden_dim)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('SNP',  'snpAssociatedWithDisease', 'Phenotype'): SAGEConv((-1, -1), hidden_dim),
                ('Gene', 'hasProteinIsoform', 'Protein'): SAGEConv((-1, -1), hidden_dim),
                ('SNP', 'haseQTL', 'Gene'): SAGEConv((-1, -1), hidden_dim),
                ('SNP', 'hasPqtl', 'Protein'): SAGEConv((-1, -1), hidden_dim),
                ('Gene', 'geneAssociatedWithDisease', 'Phenotype'): SAGEConv((-1, -1), hidden_dim),
            }, aggr='sum')
            self.convs.append(conv)

        self.lin1 = torch.nn.Linear(hidden_dim + covariate_dim, 64)
        self.lin2 = torch.nn.Linear(hidden_dim, 1) # to get one scalar logit output for confidence in positive classification

        # storage for downstream metrics calc
        self._val_hats  = []
        self._val_trues = []

    def forward(self, data):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict

        # Debug prints
        for rel, idx in data.edge_index_dict.items():
            print(f"edge_index {rel} dtype:", idx.dtype)
        for nt, bidx in data.batch_dict.items():
            print(f"batch_ptr {nt} dtype:", bidx.dtype)

        x_dict['SNP'] = self.init_proj(x_dict['SNP'])
            
        for conv in self.convs:
            out_dict = conv(x_dict, edge_index_dict)
            # preserve any node‐types that weren’t written by this conv: (SNP nodes bc not destination)
            for node_type, x in x_dict.items():
                if node_type not in out_dict:
                    out_dict[node_type] = x
            x_dict = {k: x.relu() for k, x in out_dict.items()}

        batch_dict = data.batch_dict  # Needed for pooling

        # pooling over SNP node type
        graph_embed = global_mean_pool(x_dict['SNP'], batch_dict['SNP'])  # shape [batch_size, hidden_dim]

        B, H = graph_embed.size()
        print(B, H) # batch size and hidden dimension

        cov = data.patient_covariates
        print(cov)
        if isinstance(cov, (list, tuple)):
            covariates = torch.stack(cov, dim=0)  # → [B, 6]
        # already a 2-D tensor [B,6]
        elif cov.dim() == 2 and cov.size(0) == B:
            covariates = cov
        # a single flat 1-D tensor of length B*6
        elif cov.dim() == 1 and cov.numel() == B * 6:
            covariates = cov.view(B, 6)

        # debug:
        print("graph_embed:", graph_embed.shape, "covariates:", covariates.shape)

        covariates = covariates.to(graph_embed.device).float()

        x = torch.cat([graph_embed, covariates], dim=1)

        x = F.relu(self.lin1(x))
        x = torch.sigmoid(self.lin2(x)).squeeze(1)  # shape [batch_size]
        return x

    def training_step(self, batch, batch_idx):
        y_hat = self.forward(batch)
        loss = F.binary_cross_entropy(y_hat, batch.y.float())
        self.log('train_loss', loss, batch_size=batch.y.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        y     = batch.y.float()
        y_hat = self.forward(batch)

        self._val_trues.append(y.detach().cpu().numpy())
        self._val_hats.append(y_hat.detach().cpu().numpy())

    def on_validation_epoch_end(self):
        # concatenate all batches
        y_true = np.concatenate(self._val_trues, axis=0)
        y_prob = np.concatenate(self._val_hats, axis=0)
        y_pred = (y_prob > 0.5).astype(int)

        # point‐estimates
        auc0   = roc_auc_score(y_true, y_prob)
        acc0   = accuracy_score(y_true, y_pred)
        f10    = f1_score(y_true, y_pred)
        auprc0 = average_precision_score(y_true, y_prob)

        # bootstrap for std
        auc_mean,   auc_std   = bootstrap_metric(y_true, y_prob,    roc_auc_score)
        acc_mean,   acc_std   = bootstrap_metric(y_true, y_pred,    accuracy_score, is_prob=False)
        f1_mean,    f1_std    = bootstrap_metric(y_true, y_pred,    f1_score, is_prob=False)
        auprc_mean, auprc_std = bootstrap_metric(y_true, y_prob,    average_precision_score)

        self.log('val_auc',   auc0,   prog_bar=True)
        self.log('val_auc_std',   auc_std)
        self.log('val_acc',   acc0,   prog_bar=True)
        self.log('val_acc_std',   acc_std)
        self.log('val_f1',    f10,    prog_bar=True)
        self.log('val_f1_std',    f1_std)
        self.log('val_auprc', auprc0, prog_bar=True)
        self.log('val_auprc_std', auprc_std)

        # clear for next epoch
        self._val_trues.clear()
        self._val_hats.clear()


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

# Load dataset
dataset = PatientGraphDataset("data/patient_graphs/")

# ---------------- used this to calculate appropriate threshold for probs -> patient preds using cross validation
# import numpy as np
# from sklearn.model_selection import StratifiedKFold, cross_val_predict

# all_graphs = list(dataset)
# all_labels = np.array([g.y.item() for g in all_graphs])

# class SklearnWrapper(BaseEstimator, ClassifierMixin):
#     def __init__(self, num_layers=2, covariate_dim=6, max_epochs=10, batch_size=32, device='cpu'):
#         # store all hyperparameters here
#         self.num_layers    = num_layers
#         self.covariate_dim = covariate_dim
#         self.max_epochs    = max_epochs
#         self.batch_size    = batch_size
#         self.device        = device

#         # instantiate the actual LightningModule
#         self.model = HeteroGraphClassifier(
#             num_layers=self.num_layers,
#             covariate_dim=self.covariate_dim
#         )

#     def fit(self, X, y):
#         loader  = DataLoader(X, batch_size=32, shuffle=True)
#         trainer = pl.Trainer(max_epochs=10, logger=False, enable_checkpointing=False)
#         trainer.fit(self.model, loader)

#         if y is None:
#             y = np.array([g.y.item() for g in X])
#         self.classes_ = np.unique(y)
    
#         return self

#     def predict_proba(self, X):
#         loader = DataLoader(X, batch_size=32)
#         self.model.eval()
#         all_p = []
#         with torch.no_grad():
#             for batch in loader:
#                 p = self.model(batch).cpu().numpy()
#                 all_p.append(p)
#         all_p = np.concatenate(all_p)
#         # return shape (n_samples, 2)
#         return np.vstack([1-all_p, all_p]).T
    
#     def predict(self, X):
#         # return the positive-class probability as “prediction”
#         probs = self.predict_proba(X)
#         return probs[:, 1]

# # 3) **Here is where you call cross_val_predict**, to get out‑of‑fold scores:
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# clf = SklearnWrapper(
#     num_layers=2,
#     covariate_dim=6,
#     max_epochs=10,
#     batch_size=32,
#     device='cpu'
# )
# oof_proba = cross_val_predict(
#     clf,
#     all_graphs,
#     all_labels,
#     cv=skf,
#     method="predict_proba",
#     n_jobs=1
# )[:,1]   # take the “positive” class probability

# # 4) Tune your threshold on `oof_proba` vs. `all_labels`
# from sklearn.metrics import precision_recall_curve
# prec, rec, thr = precision_recall_curve(all_labels, oof_proba)
# f1s = 2*prec*rec/(prec+rec+1e-8)
# best_thr = thr[np.argmax(f1s)]
# print(f"CV‑tuned threshold = {best_thr:.3f}")

# ----------------

# Split into train/val/test as needed
train_dataset = dataset[:int(0.7 * len(dataset))]
val_dataset = dataset[int(0.7 * len(dataset)):]

print("split dataset into train and val sets")

# Use PyG's DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=16, num_workers=0)

print("created dataloaders for train and val sets")

# Read metadata from a sample graph
sample_graph = train_dataset[0]
covariate_dim = sample_graph.patient_covariates.shape[0]
print(f"Covariate dimension: {covariate_dim}")
# just loading correct covariate dimension from the first graph in the dataset (will be the same for all graphs)

# Instantiate model
model = HeteroGraphClassifier(num_layers=2, covariate_dim=covariate_dim)

print("created model")
# Train
trainer = pl.Trainer(
    max_epochs=10,
    accelerator='auto',
    log_every_n_steps=10,
    deterministic=True
)

print("created trainer, starting training...")
trainer.fit(model, train_loader, val_loader)

print("training complete, getting validation results")
results = trainer.validate(model=model, dataloaders=val_loader)
print(results)

# getting patient IDs and corresponding predictions from model (from validation set, and training set)
model.eval()
records = []

with torch.no_grad():
    for batch in val_loader:                     
        # model scores
        scores = model(batch).cpu()                # shape [B], floats in [0,1]
        # patient IDs
        ids = batch.patient_id                        # list or tensor of length B
        # true labels
        trues = batch.y.cpu()

        # 4) collect
        for pid, score, true in zip(ids, scores, trues):
            records.append({
                "PMBB_ID": pid,
                "score": float(score),
                "true_label": int(true),
                "status" : 'val'
            })
    for batch in train_loader:                       
        # model scores
        scores = model(batch).cpu()                # shape [B], floats in [0,1]
        # patient IDs
        ids = batch.patient_id                        # list or tensor of length B
        # true labels
        trues = batch.y.cpu()

        for pid, score, true in zip(ids, scores, trues):
            records.append({
                "PMBB_ID": pid,
                "score": float(score),
                "true_label": int(true),
                "status" : 'train'
            })

# recording patient predictions for this run
df = pd.DataFrame(records)
df.to_csv("patient_predictions.csv", index=False)
print("Wrote", len(df), "rows to patient_predictions.csv")