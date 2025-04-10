# Graph Neural Network for Nucleic Acid Binding Site Prediction

## Overview

This project focuses on predicting nucleic acid binding sites (dsDNA, ssDNA, and RNA) on proteins using Graph Neural Networks (GNNs). The model leverages both structural and sequence-based information to learn residue-level binding propensities, enabling fine-grained classification of binding regions.

## Dataset Description

The dataset originates from prior bachelor's research and contains:

- 3997 dsDNA-binding proteins  
- 350 ssDNA-binding proteins  
- 809 RNA-binding proteins  

Protein structures were retrieved using PPI3D, with nucleic acid residues excluded. Each protein contains only amino acid residues, but still retains its native binding interface.

## Data Preparation

### Structural Representation

Protein structures are converted into graph representations using [this repository](https://github.com/kliment-olechnovic/generating-graphs-of-protein-receptors), which performs Voronoi tessellation on atomic coordinates and outputs:

- Nodes: Each row corresponds to an atom with features such as:
  - `residue_type`, `sas_area`, `voromqa_score`, `volume`, `coordinates`
- Edges: Each row defines atomic contacts, including:
  - `contact_area`, `inter-atomic_distance`, `voromqa_energy`

These features are aggregated to the residue level, where:

- Surface area and volume are summed
- Coordinates are averaged
- Labels are binary (binding vs non-binding) or multi-class (dsDNA, ssDNA, RNA)

### Sequence Representation

Residue-level embeddings are extracted using ESM2, a Protein Language Model (PLM). Each residue receives a 320-dimensional embedding vector. These are merged with structural features using the residue indices.

## Graph Construction

Each protein is encoded as a graph:

```
G = (X, E_index, E_feat, y)
```

Where:

- X ∈ ℝⁿˣ³²⁹ – node feature matrix with:
  - 9 structural features (e.g., `sas_area`, `volume`, etc.)
  - 320-dimensional ESM2 sequence embeddings
- E_index ∈ ℕ²ˣᵉ – edge index (connectivity)
- E_feat ∈ ℝᵉˣ² – edge features (`contact_area`, `distance`)
- y ∈ {0,1}ⁿ – binary or multi-class label for each residue
