import os
import torch
from torch_geometric.data import Dataset, HeteroData

class PatientGraphDataset(Dataset):
    def __init__(self, graph_dir, transform=None, pre_transform=None, filter_bad_labels=True):
        """
        Args:
            graph_dir (str): Path to directory containing patient .pt files
            filter_bad_labels (bool): Whether to exclude unlabeled (-1) graphs
        """
        super().__init__(graph_dir, transform, pre_transform)
        self.graph_dir = graph_dir
        self.file_names = sorted([
            f for f in os.listdir(graph_dir) if f.endswith('.pt')
        ])
        
        # filter out patients with missing label (y = -1)
        if filter_bad_labels:
            self.file_names = [
                f for f in self.file_names
                if torch.load(os.path.join(graph_dir, f), weights_only=False).y.item() != -1
            ]

    def len(self):
        return len(self.file_names)

    def get(self, idx):
        file_path = os.path.join(self.graph_dir, self.file_names[idx])
        graph = torch.load(file_path, weights_only=False, map_location='cpu')
        
        # safety checks
        assert isinstance(graph, HeteroData), "Graph is not a HeteroData object"
        assert hasattr(graph, 'y'), "Missing label in graph"
        assert hasattr(graph, 'patient_covariates'), "Missing covariates in graph"

        return graph
