# Dataset and dataloader definitions

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import h5py
import json
import numpy as np
from utils import bp2matrix

class EmbeddingDataset(Dataset):
    def __init__(self, dataset_path, embedding_name, probing_path):#), beta):
        # Loading dataset
        data = pd.read_csv(dataset_path)
        self.sequences = data.sequence.tolist()
        self.ids = data.id.tolist()

        # Loading representations
        self.embeddings = {}
        try:
            embeddings = h5py.File(f"data/embeddings/{embedding_name}_ArchiveII.h5", "r")
        except FileNotFoundError:
            print(f"Embedding file not found: {embedding_name}")
            raise

        # Loading probing
        self.probing = {}
        try:
            probing = torch.load(probing_path)
        except FileNotFoundError:
            print(f"Probing file not found: {probing_path}")
            raise
            
        # Keep only sequences in dataset_path
        for seq_id in self.ids:
            embedding = torch.from_numpy(embeddings[seq_id][()])
            # probing_reshaped = probing[seq_id].reshape(probing[seq_id].shape[0], 1)
            # probing_reshaped = (1-beta)*probing_reshaped + beta*np.random.uniform(0, 1, probing_reshaped.shape)
            # embedding = torch.from_numpy(np.hstack([embedding, probing_reshaped])) # L x d -> L x d+1
            self.embeddings[seq_id] = embedding
            self.probing[seq_id] = probing[seq_id]
        self.base_pairs = [
            json.loads(data.base_pairs.iloc[i]) for i in range(len(data))
        ]
        
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        seq_id = self.ids[idx]
        sequence = self.sequences[idx]

        if seq_id not in self.embeddings:
            print(f"{seq_id} not present")
            raise KeyError(f"Sequence ID {seq_id} not found in embeddings")
            
        seq_emb = self.embeddings[seq_id]
        L = len(sequence)

        Mc = bp2matrix(L, self.base_pairs[idx])
        return {
            "seq_id": seq_id, 
            "seq_emb": seq_emb, 
            "contact": Mc, 
            "probing": self.probing[seq_id],
            "L": L, 
            "sequence": sequence
        }

def pad_batch(batch):
    """Collate function to pad batches of variable length sequences"""
    seq_ids, seq_embs, Mcs, probings, Ls, sequences = [
        [batch_elem[key] for batch_elem in batch] 
        for key in ["seq_id", "seq_emb", "contact", "probing", "L", "sequence"]
    ]
    
    embedding_dim = seq_embs[0].shape[1]  # seq_embs is a list of tensors of size L x d
    batch_size = len(batch)
    max_L = max(Ls)
    
    seq_embs_pad = torch.zeros(batch_size, max_L, embedding_dim)
    # cross entropy loss can ignore the -1s
    Mcs_pad = -torch.ones((batch_size, max_L, max_L), dtype=torch.long)
    probings_pad = torch.zeros(batch_size, max_L)#, 1)

    for k in range(batch_size):
        seq_embs_pad[k, : Ls[k], :] = seq_embs[k][:Ls[k], :]
        Mcs_pad[k, : Ls[k], : Ls[k]] = Mcs[k]
        probings_pad[k, : Ls[k]] = probings[k][:Ls[k]]
        
    return {
        "seq_ids": seq_ids, 
        "seq_embs_pad": seq_embs_pad, 
        "contacts": Mcs_pad, 
        "probings": probings_pad,
        "Ls": Ls, 
        "sequences": sequences
    }

def create_dataloader(embedding_name, partition_path, probing_path, batch_size, shuffle, collate_fn=pad_batch):
    """Create a dataloader for RNA sequences"""
    dataset = EmbeddingDataset(
        embedding_name=embedding_name,
        dataset_path=partition_path,
        probing_path=probing_path,
        # beta=beta,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
