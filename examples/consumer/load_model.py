"""Example of what loadsafe flags in consumer code (do not copy)."""
from transformers import AutoModel
import torch

# Unpinned + trust_remote_code=True -> LS210, LS211
m = AutoModel.from_pretrained("org/name", trust_remote_code=True)
# Unsafe torch.load -> LS212
state = torch.load("weights.pt")
