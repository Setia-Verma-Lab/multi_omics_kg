# step 2: putting patient data (copies of knowledge graph loaded) into a pyg graph object
# Builds per-patient PyG graphs

# data needed: patient covariates (age and PCs) in one file, genotype dosage file 

#create a "copy" of the graph for each endo and control patient in PMBB. 
# The "value" of each SNP node would be their genotype. 
# You use each graph as a training or testing sample in a graph classification task (rather than a link prediction task). 
# Basically, you train a classifier that predicts "for this patient, does the graph likely represent endo or not endo?"

# value for all of the other nodes (not SNP) could be either a dummy node (e.g., assign a float "1.0" to all of them) or 
# it could be the embedding you learned. 

import torch
import pandas as pd
from torch_geometric.data import HeteroData
import os
import gzip

GRAPH_PATH = "graph_topology_hetero.pt"
GENOTYPE_DOSAGE_PATH = "PMBB-Release-2024-3.0_genetic_imputed.all_chromosomes.csv.gz"   # SNPs x Patients, genotype dosage values
COVARIATE_PATH = "patient_covariates.csv"      # Patient ID, age, PC1-5
LABEL_PATH = "chart_review_and_EHR_labels.csv"
OUTPUT_DIR = "patient_graphs/"                 # Where to save individual patient HeteroData
os.makedirs(OUTPUT_DIR, exist_ok=True)

# shared base graph
base_graph = torch.load(GRAPH_PATH, weights_only=False)
# print(base_graph['SNP'].to_dict().keys())

# covariate_df: index = patient ID, columns = age, PCs 1-5
covariate_df = pd.read_csv(COVARIATE_PATH, index_col=0)

# labels (index = patient ID, column = label)
label_df = pd.read_csv(LABEL_PATH, index_col=0)

# pre-exported SNP IDs
snp_ids = pd.read_csv('snp_ids.csv')['snp_id'].tolist()

print("Starting graph generation...")

with gzip.open(GENOTYPE_DOSAGE_PATH, "rt") as f:
    header_line = next(f).strip()

# assuming your first column is something like "snp_id" or "Unnamed: 0"
cols = header_line.split("\t")
patient_ids = cols[1:]    # drop the first column name

# loop over each patient and build graph
for patient_id in patient_ids:
    print(f"Processing patient: {patient_id}")

    # patient-specific SNP dosage features
    # Stream just the single column for this patient (plus SNP index)
    dosage_df = pd.read_csv(
        GENOTYPE_DOSAGE_PATH,
        usecols=['SNP_ID', patient_id],
        compression='gzip',
        delimiter='\t'
    )
    dosage_series = dosage_df[patient_id]

    # align SNP dosage to SNP order in base graph
    snp_dosages = torch.tensor([
        dosage_series.get(snp_id, 0.0) for snp_id in snp_ids
    ], dtype=torch.float).unsqueeze(1)

    # if want to combine embedding and dosage for snp node:
    # snp_features = torch.cat([snp_embedding, snp_dosage], dim=1) 

    # cpy the base graph
    patient_graph = base_graph.clone()
    patient_graph['SNP'].x = snp_dosages

    # patient-level age and PCs
    if patient_id in covariate_df.index:
        covariates = torch.tensor(covariate_df.loc[patient_id].values, dtype=torch.float)
    else:
        print(f"Warning: No covariates for {patient_id}, using zeros")
        covariates = torch.zeros(covariate_df.shape[1])

    patient_graph.patient_covariates = covariates
    patient_graph.patient_id = patient_id  # for traceability

    # case control label
    if patient_id in label_df.index:
        patient_graph.y = torch.tensor([label_df.loc[patient_id, 'Chart_Adeno_or_Endo']], dtype=torch.long)
    else:
        print(f"⚠️ Missing label for {patient_id}; using -1")
        patient_graph.y = torch.tensor([-1], dtype=torch.long)

    torch.save(patient_graph, os.path.join(OUTPUT_DIR, f"{patient_id}.pt"))

print(f"finished building patient graphs.")

# to sanity check resulting graph format: 
# graph = torch.load("patient_graphs/0_PMBB9084826138347.pt", weights_only=False)

# # type and contents
# print(graph)
# print("Node types:", graph.node_types)
# print("Edge types:", graph.edge_types)

# # per-node features
# for ntype in graph.node_types:
#     x = graph[ntype].x
#     print(f"  {ntype:10s}.x shape = {tuple(x.shape)}")

# # check SNP dosage
# print("SNP dosage column shape:", graph["SNP"].x.size())

# # SNP dosage tensor
# dosages = graph["SNP"].x.squeeze()     # shape [#SNPs]
# print("SNP dosage vector shape:", dosages.shape)

# # patient covariates and label
# print("Covariates:", getattr(graph, "patient_covariates", None))

# pid = graph.patient_id  # should be "PMBB1068409755053"
# # cov_series = covariate_df.loc[pid]

# # your tensor is:
# cov_tensor = graph.patient_covariates
# print(cov_tensor)
# print(cov_tensor.dim())

# # print a side‐by‐side view:
# print(f"\nSample covariates for patient {pid}:")
# for val in cov_tensor.tolist():
#     print(f"  {val:.4f}")

# print("Covariate vector shape:", graph.patient_covariates.shape)
# print("Label (y):", graph.y)