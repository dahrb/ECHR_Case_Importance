"""
echr.graph — Temporal Graph Network (TGN) over ECHR citation graphs.

Responsibilities
----------------
- Build node/edge datasets from outcome_cases citation metadata
  (extractedappno / sclappnos fields → temporal citation edges)
- Train TGN model on citation graph per article
- At inference: compute node similarity scores for KG-augmented prediction
- Expose pre-trained node embeddings for use in echr.experiments

Inputs
------
  data/network/article{N}/network_nodes.csv
  data/network/article{N}/network_edges.csv

Outputs
-------
  data/graph/article{N}/best_model.pth   — trained TGN weights
  data/graph/article{N}/embeddings.pkl   — node embeddings for inference

Current implementation:
  old/Knowledge_Graph/network_TGN.py     — TGN model definition + training loop
  old/Knowledge_Graph/data_process_KG.py — ECHRData dataset class (nodes + edges → PyG)
Migration: parametrise by article, move Model + ECHRData classes here.
"""
