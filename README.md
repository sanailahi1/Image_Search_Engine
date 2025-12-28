# Image Search Engine using OpenCLIP + FAISS

This project implements a **content-based image search engine** using modern vision-language embeddings and efficient vector search.  
The goal was not just to make it work, but to design a **scalable, memory-aware, and explainable pipeline** from raw images to semantic retrieval.

---

## Overview

The system takes an image as a query and retrieves visually and semantically similar images from a dataset.  
It is built around three core ideas:

- **Strong representations** (OpenCLIP embeddings)
- **Efficient similarity search** (FAISS)
- **Practical engineering choices** (batch processing, clean data splits, evaluation)

Dataset used: **101 Object Categories** (≈9k images, 100+ classes)

---

## High-Level Pipeline

1. Dataset download and structured exploration  
2. Image preprocessing using OpenCLIP transforms  
3. Batch-wise embedding generation (memory safe)  
4. Vector indexing with FAISS  
5. Query-time similarity search  
6. Quantitative evaluation (Precision, Recall, mAP)  
7. Embedding space visualization (PCA, t-SNE)

Each step is modular and can be replaced or extended independently.

---

## Key Design Decisions

### Why OpenCLIP?
- Produces **semantic embeddings**, not just pixel-level features
- Generalizes well without task-specific training
- Keeps the system close to real-world retrieval setups

### Why FAISS?
- Optimized for **large-scale vector similarity search**
- Fast and reliable for L2-based nearest neighbor retrieval
- Clean separation between feature extraction and search

### Batch Processing
Instead of loading the full dataset at once, embeddings are generated in batches to:
- Avoid Colab crashes
- Reduce GPU/CPU memory pressure
- Make the pipeline scalable to larger datasets

---

## Dataset Handling

- Images are loaded dynamically using directory traversal
- Labels are inferred directly from folder structure
- Dataset is split into:
  - **80% indexing set**
  - **20% query set**

This avoids data leakage and allows proper evaluation.

---

## Embedding Visualization

To verify that the learned embeddings are meaningful, dimensionality reduction is applied:

- **PCA** for global variance structure
- **t-SNE (2D & 3D)** for local neighborhood analysis

The visualizations confirm that semantically similar images cluster together in embedding space.

---

## Evaluation Strategy

Retrieval quality is measured using standard IR metrics:

- Precision@K
- Recall@K
- Mean Average Precision (mAP)

Ground truth relevance is defined using class labels, allowing objective evaluation of retrieval performance.

---

## Tech Stack

- Python
- PyTorch
- OpenCLIP
- FAISS
- NumPy
- scikit-learn
- Matplotlib
- KaggleHub

---

## What This Project Demonstrates

- Understanding of **representation learning**
- Practical use of **vector databases**
- Ability to design **end-to-end ML pipelines**
- Awareness of **memory and compute constraints**
- Clean separation between data, model, indexing, and evaluation

This project is intentionally structured like a real system, not a notebook experiment.

---

## Possible Extensions

- Text-to-image search using CLIP text embeddings
- Approximate FAISS indexes for larger datasets
- Web or API-based query interface
- Persisted vector storage
- Cross-modal retrieval

---

## Author

**Sana Ilahi**  
BSIT Student | Machine Learning & Systems Enthusiast

---

