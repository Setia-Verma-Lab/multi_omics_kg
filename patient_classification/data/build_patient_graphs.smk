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
import gc

GRAPH_PATH = "graph_topology_hetero.pt"
GENOTYPE_DOSAGE_PATH = "final_dosage_matrix.traw"   # SNPs x Patients, dosage values
COVARIATE_PATH = "patient_covariates.csv"      # Patient ID, age, PC1-5
LABEL_PATH = "chart_review_and_EHR_labels.csv"
OUTPUT_DIR = "patient_graphs/"                 # Where to save individual patient HeteroData
SNP_IDS_CSV = 'snp_ids.csv'

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Starting graph generation...")

with open(GENOTYPE_DOSAGE_PATH, "rt") as f: header_line = next(f).strip()

cols = header_line.split("\t")
PATIENT_IDS = [col for col in cols if col.startswith('0_')]

rule all:
    input: 
        expand(os.path.join(OUTPUT_DIR, "{patient_id}.pt"), patient_id = PATIENT_IDS) ## replace with PATIENT_IDS
        # 'final_dosage_matrix.traw',
        # 'snps_to_drop.csv'

rule update_dosage_matrix:
    input:
        genotype = 'endo_dosage_matrix.traw',
        snp_ids = SNP_IDS_CSV
    output:
        updated_matrix = GENOTYPE_DOSAGE_PATH,
        snps_to_drop = 'snps_to_drop.csv'
    resources: mem_mb=16000
    run:
        df = pd.read_csv(input.genotype, sep='\t')

        # add SNP_ID column to dosage_df: CHR_POS_REF
        df['SNP_ID'] = df['CHR'].astype(str) + '_' + df['POS'].astype(str) + '_' + df['COUNTED'].astype(str)
        df['CHR_POS_ID'] = df['CHR'].astype(str) + '_' + df['POS'].astype(str)

        # store non-dosage info 
        metadata_cols = ['CHR', 'SNP', '(C)M', 'POS', 'COUNTED', 'ALT', 'SNP_ID', 'CHR_POS_ID']
        dosage_cols = [col for col in df.columns if col not in metadata_cols]

        # flip genotype values
        df[dosage_cols] = 2.0 - df[dosage_cols]
        # get mean dosage per snp
        df['mean_dosage'] = df[dosage_cols].mean(axis=1, skipna=True)

        # sort dosage_df by snp_means ascending=False (to keep the allele dosage thats most common)
        df.sort_values(by='mean_dosage', ascending=False, inplace=True)
        # drop duplicate IDs (keep='first')
        df.drop_duplicates(subset='SNP_ID', keep='first', inplace=True)
        # fill in NA values with mean dosage (for each row, across all participants)
        df[dosage_cols] = df[dosage_cols].apply(lambda row: row.fillna(df.loc[row.name, 'mean_dosage']),axis=1)

        # ------------------------------------------------- 

        # merging with SNP IDs from KG, handling zero matches/multi allelic SNPs/ref alt allele flips
        snp_ids_df = pd.read_csv(input.snp_ids) # 752,891 rows
        # dosage_df = pd.read_csv(input.mean_dosage_matrix, sep='\t')
        merged_df = pd.merge(snp_ids_df, df, how='inner', on='SNP_ID', suffixes=('', '_dosage')) # 746,880 rows (drop SNPs from KG that only have multi allelic match in PMBB data)
        # save new dosage matrix
        merged_df.to_csv(output.updated_matrix, sep='\t', index=False)
        
        # del merged_df

        drop_snps_df = pd.merge(snp_ids_df, df, how='left', on='SNP_ID', suffixes=('', '_dosage'))
        # getting SNPs that are in the KG but NOT in PMBB genotype data
        unmatched = drop_snps_df[drop_snps_df.filter(like='0_PMBB').isnull().all(axis=1)]

        unmatched[['CHR_POS_ID']].to_csv(output.snps_to_drop, index=False)
        # takes care of ever having multiple ID matches, bc highest avg dosage will automatically have been selected

rule make_patient_graphs:
    input:
        graph       = GRAPH_PATH,
        genotype    = GENOTYPE_DOSAGE_PATH,
        covariates  = COVARIATE_PATH,
        labels      = LABEL_PATH
    output:
        os.path.join(OUTPUT_DIR, "{patient_id}.pt")
    resources:
        mem_mb=16000
    run:
        # shared base graph
        base_graph = torch.load(input.graph, weights_only=False)

        # covariate_df: index = patient ID, columns = age, PCs 1-5
        covariate_df = pd.read_csv(input.covariates, index_col=0)
        covariate_df.drop(['person_id', 'SEX'], axis=1, inplace=True)

        # Load labels (index = patient ID, column = label)
        label_df = pd.read_csv(input.labels, index_col=0)

        # pre-exported SNP IDs: need to have chr, pos, ref and alt allele list
        # snp_ids_df = pd.read_csv(input.snp_ids)
        
        pid = wildcards.patient_id
        pid = str(pid).split('_')[1]
        dosage_pid = '0_' + pid
        print(f"Processing patient: {pid}")

        # 2. Set patient-specific SNP dosage features
        dosage_df = pd.read_csv(input.genotype,usecols=[dosage_pid],delimiter='\t')
        print("read in dosage df")

        # Align SNP dosage to SNP order in base graph
        # then here, all i would need to do is just take the pid column and convert to a list
        dosages = dosage_df[dosage_pid].to_numpy(dtype=float) # making a float array
        snp_dosages = torch.tensor(dosages, dtype=torch.float).unsqueeze(1) # making a column vector

        print("made tensor of patient specific dosages")
    
        # if want to combine embedding and dosage for snp node:
        # snp_features = torch.cat([snp_embedding, snp_dosage], dim=1) 

        # copy the base graph
        patient_graph = base_graph.clone()
        patient_graph['SNP'].x = snp_dosages

        print("assigned snp dosages to graph")

        # add patient-level age and PCs
        if pid in covariate_df.index:
            row = covariate_df.loc[pid]
            covariates = torch.tensor(row.values, dtype=torch.float)
        else:
            print(f"Warning: No covariates for {pid}, using zeros")
            covariates = torch.zeros(covariate_df.shape[1])

        # Store as custom attribute
        patient_graph.patient_covariates = covariates
        patient_graph.patient_id = pid

        print("added covariates to graph")

        # add case control label
        if pid in label_df.index:
            raw = label_df.at[pid, 'Chart_Adeno_or_Endo']      # e.g. 1.0
            lbl = int(raw)  
            patient_graph.y = torch.tensor([lbl], dtype=torch.long)
        else:
            print(f"missing label for {pid}; using -1")
            patient_graph.y = torch.tensor([-1], dtype=torch.long)

        print("added cohort label")

        torch.save(patient_graph, output[0])
