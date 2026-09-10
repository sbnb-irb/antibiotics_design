import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from hgraph.mol_graph import MolGraph

class HAEMoleculeDatasetSafe(Dataset):
    def __init__(self, data, vocab, avocab, batch_size):
        safe_index = [_ for _ in range(len(data))]
        self.vocab = vocab
        self.avocab = avocab
        self.safe_index = safe_index
        self.batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]
        self.batches_safe_index = [self.safe_index[i : i + batch_size] for i in range(0, len(data), batch_size)]
    def __len__(self):
        return len(self.batches)
    def __getitem__(self, idx):
        _batch = MolGraph.tensorize(self.batches[idx], self.vocab, self.avocab)
        _smiles = self.batches[idx]
        _molIndex = self.batches_safe_index[idx]
        return _batch,_smiles,_molIndex

def validateSmilesForHAE(smiles,vocab):
    """
    Check if smiles items are valid for the system
    """
    data = smiles
    safe_data = []
    safe_index = []
    print("Checking if smiles are valid...")
    for _i,mol_s in tqdm(enumerate(data)):
        hmol = MolGraph(mol_s)
        ok = True
        for node,attr in hmol.mol_tree.nodes(data=True):
            smiles = attr['smiles']
            ok &= attr['label'] in vocab.vmap
            for i,s in attr['inter_label']:
                ok &= (smiles, s) in vocab.vmap
        if ok: 
            safe_data.append(mol_s)
            safe_index.append(_i)
    print(f'After pruning {len(data)} -> {len(safe_data)}')
    return safe_data,safe_index

class HAEMoleculeDatasetNew(Dataset):
    def __init__(self, data, vocab, avocab, batch_size):
        safe_data, safe_index = validateSmilesForHAE(data,vocab)
        self.batches = [safe_data[i : i + batch_size] for i in range(0, len(safe_data), batch_size)]
        self.batches_safe_index = [safe_index[i : i + batch_size] for i in range(0, len(safe_data), batch_size)]
        self.vocab = vocab
        self.avocab = avocab
        self.safe_index = safe_index
        self.safe_data = safe_data

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        _batch = MolGraph.tensorize(self.batches[idx], self.vocab, self.avocab)
        _smiles = self.batches[idx]
        _molIndex = self.batches_safe_index[idx]
        return _batch,_smiles,_molIndex
        
class LatentsDataset(Dataset):
  def __init__(self, latents,device="cuda:0",max_items=-1):
      self.latents = latents
      if max_items>0:
        self.latents = self.latents[:max_items,:]
      self.device = device
  def __len__(self):
      return self.latents.shape[0]
  def to_tensor(self,ix):
    return torch.tensor(ix.flatten(),device=self.device,dtype=torch.float32)
  def __getitem__(self, idx):
    x = self.latents[idx]
    return self.to_tensor(x)