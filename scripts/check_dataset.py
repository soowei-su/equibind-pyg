"""
Validation script for the PDBBindPyG data pipeline (Phase 1).

This script performs the following checks:
1.  Initializes the `PDBBindPyG` dataset class.
2.  Forces reprocessing to test the `process()` method.
3.  Logs the number of successfully processed and failed complexes.
4.  Fetches and prints the first sample (`dataset[0]`) to verify its structure.
5.  Initializes a `DataLoader`.
6.  Attempts to load and print one batch to validate the `collate_fn`.

Successful execution of this script validates that the entire data pipeline
is working correctly, from raw files to model-ready batches.
"""
import torch
import logging
from torch_geometric.loader import DataLoader
from equibind_pyg.data.pdbbind_dataset import PDBBindPyG

# --- Configuration ---
# Define the paths based on our project structure
ROOT_DIR = "data/pyg_data"  # This is where processed .pt files will be saved
ID_FILE = "data/dev_ids.txt" # The list of 100 IDs to process
BATCH_SIZE = 8
K_NEIGHBORS = 30 # The 'k' for the receptor k-NN graph

# Set to True to clear the 'processed/' directory and re-run all parsing.
# Useful for testing changes in `preprocess.py` or `pdbbind_dataset.py`.
FORCE_REPROCESS = True

# Set up logging so we can see the progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger()

log.info("--- Starting Data Pipeline Validation ---")

def validate_dataset_and_loader():
    """
    Initializes the dataset and dataloader, logging progress and outputs.
    """
    log.info(f"Initializing PDBBindPyG dataset...")
    log.info(f"  Root dir: {ROOT_DIR}")
    log.info(f"  ID file: {ID_FILE}")
    log.info(f"  Force Reprocess: {FORCE_REPROCESS}")
    
    try:
        # 1. Initialize the Dataset
        # This will call .process() if 'data/pyg_data/processed/' is empty
        # or if FORCE_REPROCESS is True.
        dataset = PDBBindPyG(
            root=ROOT_DIR,
            id_file=ID_FILE,
            k_receptor_neighbors=K_NEIGHBORS,
            force_reprocess=FORCE_REPROCESS
        )
    except Exception as e:
        log.error(f"Failed to initialize dataset: {e}", exc_info=True)
        return

    log.info(f"Dataset initialization complete.")
    # This count reflects only the *successfully* processed files.
    log.warning(f"  Total successfully processed complexes: {len(dataset)}")

    if len(dataset) == 0:
        log.error("Dataset is empty. Processing may have failed for all complexes.")
        log.error("Check 'data/v2020/' paths and 'preprocess.py' logs.")
        return

    # 2. Print info about the first sample
    try:
        # Fetch the first successfully processed sample
        first_sample = dataset[0]
        log.info(f"\n--- First Sample (PDB ID: {first_sample.pdb_id}) ---")
        # Print the HeteroData object structure
        print(first_sample)
        log.info("----------------------------------")
    except Exception as e:
        log.error(f"Failed to get first sample from dataset: {e}", exc_info=True)
        return

    # 3. Initialize the DataLoader
    log.info(f"Initializing DataLoader with batch size {BATCH_SIZE}...")
    # This will use the PyG `collate_fn` to create `HeteroDataBatch` objects
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 4. Try to load one batch
    try:
        # This is the critical test: can PyG collate the samples?
        batch = next(iter(loader))
        log.info(f"\n--- SUCCESSFULLY LOADED ONE BATCH (Batch Size: {BATCH_SIZE}) ---")
        print(batch)
        log.info("------------------------------------------")
        
        log.info("Batch attributes (should be HeteroDataBatch):")
        log.info(f"  PDB IDs in batch: {batch.pdb_id}")
        log.info(f"  Total receptor nodes: {batch['receptor'].num_nodes}")
        log.info(f"  Total ligand nodes: {batch['ligand'].num_nodes}")
        log.info(f"  Receptor node features shape: {batch['receptor'].x.shape}")
        log.info(f"  Ligand node features shape: {batch['ligand'].x.shape}")
        log.info(f"  Receptor pos shape: {batch['receptor'].pos.shape}")
        log.info(f"  Ligand pos shape: {batch['ligand'].pos.shape}")
        log.info(f"  Receptor edge_index shape: {batch['receptor', 'intra', 'receptor'].edge_index.shape}")
        log.info(f"  Ligand edge_index shape: {batch['ligand', 'intra', 'ligand'].edge_index.shape}")
        
    except Exception as e:
        log.error(f"Failed to load batch from DataLoader: {e}", exc_info=True)
        return

    log.info("\n--- Data Pipeline Validation SUCCESSFUL ---")
    log.info("Phase 1 is complete!")

if __name__ == "__main__":
    # This check ensures the script runs when you execute it directly
    validate_dataset_and_loader()