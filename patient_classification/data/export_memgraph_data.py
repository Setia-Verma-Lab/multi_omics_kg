from gqlalchemy import Memgraph
from gqlalchemy.transformations.translators.pyg_translator import PyGTranslator
import torch
import pandas as pd
import ast
import json

# step 1: connecting to memgraph data (already loaded) and translating into python
memgraph = Memgraph()
translator = PyGTranslator()
graph = translator.get_instance() # gets memgraph data and turns into heterodata object! so cool

print(type(graph))                 # <class 'torch_geometric.data.hetero_data.HeteroData'>
print(graph.node_types)
print(graph.edge_types)

# just making sure the embeddings got loaded and renamed as .x properly
for node_type in graph.node_types:
    if 'embedding' in graph[node_type]:
        graph[node_type].x = graph[node_type].embedding
        print(f"{node_type}: Feature shape = {graph[node_type].x.shape}")
        del graph[node_type].embedding

print("Sample SNP embedding vector:", graph['SNP'].x[0])
print("Sample Gene embedding vector:", graph['Gene'].x[0])

# save for building patient-specific graphs
torch.save(graph, 'graph_topology_hetero.pt')

# <class 'torch_geometric.data.hetero_data.HeteroData'>
# ['Protein', 'Gene', 'Phenotype', 'SNP']
# [('Gene', 'hasProteinIsoform', 'Protein'), ('Gene', 'geneAssociatedWithDisease', 'Phenotype'), ('SNP', 'haseQTL', 'Gene'), ('SNP', 'hasPqtl', 'Protein'), ('SNP', 'snpAssociatedWithDisease', 'Phenotype')]
# Protein: Feature shape = torch.Size([20195, 8])
# Gene: Feature shape = torch.Size([27175, 8])
# Phenotype: Feature shape = torch.Size([851, 8])
# SNP: Feature shape = torch.Size([752891, 8])
# Sample SNP embedding vector: tensor([ 0.4699, -0.7336,  0.3952,  0.3859, -0.0142,  0.5187,  0.4714, -0.2512])
# Sample Gene embedding vector: tensor([-0.7181, -0.6591,  0.6912,  0.7314,  0.6710,  0.8236, -0.6107,  0.2062])

# get Memgraph-internal IDs for all SNP nodes, cast to int
raw_positions = graph['SNP'].position.tolist()
positions = [int(x) for x in raw_positions]

# ask Memgraph for each node’s snp_id property
query = """
MATCH (s:SNP)
RETURN id(s) AS mem_id, s.snp_id AS snp_id, s.ref_allele AS ref_allele, s.alt_allele AS alt_allele, s.embedding AS embedding
ORDER BY mem_id;
"""
rows = list(memgraph.execute_and_fetch(query))

# sanity check
assert len(rows) == graph['SNP'].num_nodes, (
    f"expected {graph['SNP'].num_nodes} SNPs, but got {len(rows)} rows from Memgraph"
)

# get both snp_id and ref_allele in order
snp_ids = [row['snp_id'] for row in rows]
ref_alleles = [row['ref_allele'] for row in rows]
alt_alleles = [row['alt_allele'] for row in rows]
# embeddings = [row['embedding'] for row in rows]

snp_ids_df = pd.DataFrame({
    'CHR_POS_ID': snp_ids,
    'ref_allele': ref_alleles,
    'alt_allele': alt_alleles
})
snp_ids_df['SNP_ID'] = snp_ids_df['CHR_POS_ID'].astype(str) + '_' + snp_ids_df['ref_allele'].astype(str)

# had to apply this bc my alt alleles are stores sometimes as a list of strings and sometimes as a string which contains a list of strings
def safe_eval(val):
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return [val]  # fallback if string isn't a list
    return val

# snp_ids_df['embedding'] = snp_ids_df['embedding'].apply(lambda x: json.dumps(safe_eval(x)))
snp_ids_df['alt_allele'] = snp_ids_df['alt_allele'].apply(safe_eval)
snp_ids_df.to_csv('snp_ids.csv', index=False)

print(f"wrote {len(snp_ids)} snps")