import json
import random
import mgclient
import pandas as pd
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchmetrics import AUROC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from heapq import nlargest
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import math
import time

# load and normalize embeddings

def parse_vec(s):
    try:
        return json.loads(s)
    except:
        return []

df = pd.read_csv("embeddings.csv", converters={"vec": parse_vec})
df = df[df["vec"].map(len) == len(df["vec"].iloc[0])].reset_index(drop=True)

ids = df["id"].astype(int).tolist()
embs = torch.tensor(df["vec"].tolist(), dtype=torch.float32)
embs = F.normalize(embs, p=2, dim=1)
id_to_idx = {nid: i for i, nid in enumerate(ids)}
idx_to_id = {i: nid for nid, i in id_to_idx.items()}

# get all positive edges from Memgraph
conn = mgclient.connect(host="127.0.0.1", port=7687)
cur  = conn.cursor()
cur.execute(
    "MATCH (s:SNP)-[:snpAssociatedWithDisease]->(p:Phenotype) RETURN id(s), id(p)"
)
raw_pos = [(s, p) for s, p in cur.fetchall() if s in id_to_idx and p in id_to_idx]
pos = [(id_to_idx[s], id_to_idx[p]) for s, p in raw_pos]

# get all SNP and Phenotype indices
cur.execute("MATCH (s:SNP) RETURN id(s)")
snps = [id_to_idx[r[0]] for r in cur.fetchall() if r[0] in id_to_idx]
cur.execute("MATCH (p:Phenotype) RETURN id(p)")
phenos = [id_to_idx[r[0]] for r in cur.fetchall() if r[0] in id_to_idx]

# helper: richer feature interaction
def make_feat(h1, h2):
    return torch.cat([h1 * h2,
                      torch.abs(h1 - h2),
                      h1 + h2,
                      h1,
                      h2], dim=-1)

# ranking loss
target_margin = 1.0
class TripletDataset(Dataset):
    def __init__(self, pos_pairs, snp_idxs, pheno_idxs):
        self.pos     = pos_pairs
        self.snps    = snp_idxs
        self.phenos  = pheno_idxs
        self.pos_set = set(pos_pairs)

    def __len__(self):
        return len(self.pos)

    def __getitem__(self, idx):
        s_idx, p_pos = self.pos[idx]
        # sample a negative for this SNP
        while True:
            p_neg = random.choice(self.phenos)
            if (s_idx, p_neg) not in self.pos_set:
                break
        h_s   = embs[s_idx]
        h_pos = embs[p_pos]
        h_neg = embs[p_neg]
        f_pos = make_feat(h_s, h_pos)
        f_neg = make_feat(h_s, h_neg)
        return f_pos, f_neg

# train/val splits
random.shuffle(pos)
split = int(0.8 * len(pos))
pos_train, pos_val = pos[:split], pos[split:]
train_ds = TripletDataset(pos_train, snps, phenos)
# build a simple val set with (feat, label)
feats_val, labels_val = [], []
# positives
for s, p in pos_val:
    feats_val.append(make_feat(embs[s], embs[p]))
    labels_val.append(1)
# negatives (equal number)
neg_val = []
while len(neg_val) < len(pos_val):
    i = random.choice(snps)
    j = random.choice(phenos)
    if (i, j) not in set(pos_val):
        neg_val.append((i, j))
for s, p in neg_val:
    feats_val.append(make_feat(embs[s], embs[p]))
    labels_val.append(0)
feats_val = torch.stack(feats_val)
labels_val = torch.tensor(labels_val, dtype=torch.float32)
val_ds = TensorDataset(feats_val, labels_val)

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=256)

# LightningModule with MarginRankingLoss and AUC monitoring
class LinkRank(pl.LightningModule):
    def __init__(self, feat_dim, hidden_dim=128, lr=1e-3, weight_decay=1e-5, margin=1.0):
        super().__init__()
        self.save_hyperparameters()
        D = feat_dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(D, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim, hidden_dim//2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim//2, 1)
        )
        self.margin_loss = torch.nn.MarginRankingLoss(margin=margin)
        self.bce         = torch.nn.BCEWithLogitsLoss()
        self.auroc       = AUROC(task="binary")
        self._val_probs  = []
        self._val_labels = []
        self._train_losses = []

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def training_step(self, batch, _):
        f_pos, f_neg = batch
        s_pos = self(f_pos)
        s_neg = self(f_neg)
        target = torch.ones_like(s_pos)
        loss = self.margin_loss(s_pos, s_neg, target)
        self.log("train_loss", loss, on_epoch=True)

        # stash for epoch end
        self._train_losses.append(loss.detach())

        return loss

    def validation_step(self, batch, _):
        feats, y = batch
        logits = self(feats)
        loss   = self.bce(logits, y)
        probs  = torch.sigmoid(logits)
        self._val_probs.append(probs)
        self._val_labels.append(y)
        self.log("val_loss", loss, prog_bar=True)

    def on_train_epoch_end(self):
        # compute epoch‐level stats
        losses = torch.stack(self._train_losses)
        mean_loss = losses.mean()
        std_loss  = losses.std(unbiased=True)

        # log them
        self.log("train_loss_epoch", mean_loss, prog_bar=True)
        self.log("train_loss_epoch_std",  std_loss,  prog_bar=False)

        # clear for next epoch
        self._train_losses.clear()

    def on_validation_epoch_end(self):
        probs  = torch.cat(self._val_probs)
        labels = torch.cat(self._val_labels).long()
        auc    = self.auroc(probs, labels)
        self.log("val_auc", auc, prog_bar=True)
        self._val_probs.clear()
        self._val_labels.clear()

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", patience=3
        )
        return {
            'optimizer': opt,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_auc'
            }
        }

# train!
model = LinkRank(feat_dim=5*embs.size(1))
trainer = pl.Trainer(
    max_epochs=50,
    accelerator="auto",
    callbacks=[
        EarlyStopping(monitor="val_auc", mode="max", patience=5),
        ModelCheckpoint(monitor="val_auc", mode="max")
    ],
    log_every_n_steps=10
)
trainer.fit(model, train_loader, val_loader)

# get logged epoch‐level metrics
metrics = trainer.callback_metrics

train_loss_mean = metrics.get("train_loss_epoch")
train_loss_std  = metrics.get("train_loss_epoch_std")

print(f"Final training loss: {train_loss_mean:.4f} ± {train_loss_std:.4f}")

# compare positive vs. negative edge scores
model.eval()
with torch.no_grad():
    # sample 10 positives
    sample_pos = random.sample(pos_val, min(10, len(pos_val)))
    # sample 10 negatives not in pos
    sample_neg = []
    while len(sample_neg) < len(sample_pos):
        i = random.choice(snps)
        j = random.choice(phenos)
        if (i, j) not in set(pos):
            sample_neg.append((i, j))

    def score_pairs(pairs):
        feats = torch.stack([make_feat(embs[i], embs[j]) for i,j in pairs])
        logits = model(feats)
        return torch.sigmoid(logits).cpu().tolist()

    pos_scores = score_pairs(sample_pos)
    neg_scores = score_pairs(sample_neg)

print("\nPositive edge scores:")
for (i,j), s in zip(sample_pos, pos_scores):
    print(f"  {idx_to_id[i]} → {idx_to_id[j]}: {s:.4f}")
print("\nNegative edge scores:")
for (i,j), s in zip(sample_neg, neg_scores):
    print(f"  {idx_to_id[i]} → {idx_to_id[j]}: {s:.4f}")

print("Training and evaluation complete.")

# evaluation metrics on full validation set
model.eval()
with torch.no_grad():
    logits = model(feats_val)
    probs  = torch.sigmoid(logits).cpu().numpy()
    labels = labels_val.cpu().numpy().astype(int)

acc   = accuracy_score(labels, probs > 0.5)
prec, rec, f1, _ = precision_recall_fscore_support(labels, probs > 0.5, average='binary')
rocauc = roc_auc_score(labels, probs)

print(f"\nVAL METRICS → Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, ROC-AUC: {rocauc:.4f}")
print("Training and evaluation complete.")

# get all SNP -> phenotype associations, unsorted

print("about to get pheno")
# locate the phenotype node of interest in Memgraph
# phen_code = "HP_0030127"
phen_code = 'HP_0001513'
cur.execute(
    "MATCH (p:Phenotype {phenotype_id: $code}) RETURN id(p)",
    {"code": phen_code}
)
row = cur.fetchone()
if not row:
    raise KeyError(f"No Phenotype node with phenotype_id={phen_code}")
endo_db_id = row[0]
endo_idx   = id_to_idx[endo_db_id]

print("got pheno")

# prep the phenotype embedding once
device = model.net[0].weight.device
h_p = embs[endo_idx].to(device).unsqueeze(0)   # (1, d)

model.eval()


# compute ALL existing snp->pheno scores (unsorted)
known_idxs      = [s for s, p in pos if p == endo_idx]
existing_scores = []

with torch.no_grad():
    for s in known_idxs:
        node_id = idx_to_id[s]
        h_s     = embs[s].to(device).unsqueeze(0)      # (1, d)
        feats   = make_feat(h_s, h_p)                  # (1, 5*d)
        score   = torch.sigmoid(model(feats)).item()
        existing_scores.append((node_id, score))

# fetch RSIDs
node_ids = [nid for nid, _ in existing_scores]
cur.execute(
    "MATCH (s:SNP) WHERE id(s) IN $ids RETURN id(s), s.rsid",
    {"ids": node_ids}
)
rsid_map = {nid: rs for nid, rs in cur.fetchall()}

# print & save
print(f"\nAll EXISTING associations ({len(existing_scores)} total):")
data = []
for rank, (node_id, sc) in enumerate(existing_scores, 1):
    rsid = rsid_map.get(node_id, str(node_id))
    print(f"{rank:2d}. {rsid:<12} (node {node_id})  score={sc:.4f}")
    data.append({'rsid': rsid, 'score': round(sc,4)})

existing = pd.DataFrame(data)
existing.to_csv('existing_snp_obesity_scores.csv', index=False)

print("saved existing scores")

# calculate all NEW snp -> pheno scores (unsorted, that didn't exist before)
candidate_db_ids = [idx_to_id[s] for s in snps if s not in known_idxs]
new_scores       = []

with torch.no_grad():
    for node_id in candidate_db_ids:
        s_idx  = id_to_idx[node_id]
        h_s    = embs[s_idx].to(device).unsqueeze(0)
        # feats  = make_feat(h_s.expand_as(h_p), h_p)
        feats   = make_feat(h_s, h_p) 
        score  = torch.sigmoid(model(feats)).item()
        new_scores.append((node_id, score))

print("computed new scores")

BATCH_FETCH = 1000  # to not crash out server
out_path     = 'new_snp_obesity_scores.csv'

# output file and write header
with open(out_path, 'w') as fh:
    fh.write("rsid,score\n")

# fetch in batches
n = len(new_scores)
for i in range(0, n, BATCH_FETCH):
    batch = new_scores[i : i + BATCH_FETCH]
    batch_ids = [nid for nid, _ in batch]

    t0 = time.time()

    cur.execute("UNWIND $ids AS vid MATCH (s:SNP) WHERE id(s)=vid RETURN vid AS node_id, s.rsid AS rsid",
                {"ids": batch_ids})
    rows = cur.fetchall()
    t1 = time.time()
    print(f"Batch {i//BATCH_FETCH + 1} DB fetch took {t1-t0:.2f}s")
    # Build a map node_id -> rsid
    id2rs = {node_id: rsid for node_id, rsid in rows}

    # get lines to write
    lines = []
    for node_id, sc in batch:
        rsid = id2rs.get(node_id, str(node_id))
        lines.append(f"{rsid},{sc:.4f}\n")

    # add to CSV
    with open(out_path, 'a') as fh:
        fh.writelines(lines)

    print(f"  wrote batch {i//BATCH_FETCH + 1} / {math.ceil(n/BATCH_FETCH)}")

    conn.commit()

print("Done writing new scores.")