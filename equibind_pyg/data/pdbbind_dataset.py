"""
PyTorch Geometric `Dataset` class for the PDBBind dataset.

This class handles:
- Reading PDB IDs from a file.
- Finding raw .pdb and .sdf files.
- Calling the `preprocess.py` module to parse and featurize complexes.
- Assembling `HeteroData` objects according to the schema.
- Saving processed data to disk (`.pt` files).
- Loading processed data for use in a `DataLoader`.
- Robustness to parsing failures (skips and logs bad complexes).
- Resumability (skips already-processed files).
"""

import torch
import logging
from pathlib import Path
from tqdm import tqdm
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.nn import knn_graph

# Import our preprocessor functions
from equibind_pyg.data import preprocess as pp 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PDBBindPyG(Dataset):
    """
    PyTorch Geometric Dataset class for PDBBind.
    
    This class handles processing raw PDB/SDF files into
    HeteroData objects and saving/loading them from disk.
    
    The schema is defined in 'v1_rigid_spec.md'.
    """
    def __init__(self, root: str, id_file: str, k_receptor_neighbors: int = 30, 
                 force_reprocess: bool = False,
                 transform=None, pre_transform=None, pre_filter=None):
        """
        Args:
            root (str): Root directory where the dataset's 'processed/' 
                        folder will be created (e.g., 'data/pyg_data').
            id_file (str): Path to the text file containing PDB IDs 
                           (e.g., 'data/dev_ids.txt').
            k_receptor_neighbors (int): The 'k' for k-NN graph on receptor residues.
            force_reprocess (bool): If True, delete all processed files and start over.
            transform: PyG transforms (applied on-the-fly during `get()`).
            pre_transform: PyG transforms (applied before saving to disk in `process()`).
            pre_filter: PyG filters (applied before saving to disk in `process()`).
        """
        self.id_file = id_file
        self.k_receptor_neighbors = k_receptor_neighbors
        
        # This assumes 'root' is 'data/pyg_data' and the raw data 
        # (e.g., 'v2020', 'index') is in the parent 'data/' directory.
        self.raw_data_dir = Path(root).parent 
        
        # This list will hold the PDB IDs that were *successfully* processed
        # and are present in the 'processed/' directory.
        self.processed_ids = []
        
        # --- Initialization Order ---
        # 1. Manually set self.root. This allows `self.processed_dir` to be
        #    a valid path *before* calling super().__init__().
        self.root = root
        
        # 2. If `force_reprocess` is True, we manually delete the contents
        #    of the 'processed/' directory *before* PyG checks its existence.
        if force_reprocess and Path(self.processed_dir).exists():
            logging.warning(f"force_reprocess=True. Deleting all files in {self.processed_dir}")
            deleted_count = 0
            for f in Path(self.processed_dir).glob("*.pt"):
                f.unlink()
                deleted_count += 1
            logging.warning(f"Deleted {deleted_count} cached files.")

        # 3. Call the parent constructor.
        #    PyG will check if `self.processed_dir` is empty or missing files.
        #    If it is (which we just ensured if `force_reprocess=True`),
        #    it will *automatically* call `self.process()`.
        super().__init__(root, transform, pre_transform, pre_filter) 
        
        # 4. After `super()` (and potentially `self.process()`) are done,
        #    the 'processed/' folder is now populated. We must scan it
        #    to create the final, accurate list of *successfully* processed IDs.
        #    This makes the dataset robust to any failures during `process()`.
        all_files = Path(self.processed_dir).glob("*.pt")
        self.processed_ids = sorted([
            f.stem for f in all_files 
            # Filter out PyG's internal tracking files
            if f.name not in ['pre_filter.pt', 'pre_transform.pt']
        ])
        # --- End of Initialization ---

    @property
    def raw_dir(self):
        # The directory where 'v2020' (the raw data) is located.
        return self.raw_data_dir

    @property
    def raw_file_names(self):
        # This list is used by PyG to check if `process()` needs to run.
        # We only need to check for the existence of our ID file.
        if not Path(self.id_file).exists():
            raise FileNotFoundError(f"ID file not found: {self.id_file}")
        # Return the *name* of the id_file. PyG just checks if this file exists
        # in `self.raw_dir` (or in this case, the root).
        return [Path(self.id_file).name] 

    @property
    def processed_file_names(self):
        # This method tells PyG what files *should* exist *after* processing.
        # If any are missing, `process()` is called.
        # We read the id_file to get the full list of IDs we *expect* to process.
        with open(self.id_file, 'r') as f:
            pdb_ids = [line.strip() for line in f]
        # PyG will check for `self.processed_dir / f"{pdb_id}.pt"` for each ID.
        return [f"{pdb_id}.pt" for pdb_id in pdb_ids]

    def len(self):
        # The "length" of the dataset is the number of files *successfully* processed,
        # not the number of IDs in the `id_file`.
        return len(self.processed_ids)

    def get(self, idx):
        # Load the .pt file corresponding to the index
        # `self.processed_ids` is our index-to-file-stem mapper.
        pdb_id = self.processed_ids[idx]
        file_path = Path(self.processed_dir) / f"{pdb_id}.pt"
        
        # We must set `weights_only=False` (the default in older torch)
        # because we are loading a complex `HeteroData` object, not just
        # model weights.
        try:
            data = torch.load(file_path, weights_only=False)
        except AttributeError:
             # Fallback for older PyTorch versions
             data = torch.load(file_path)
        
        return data

    def process(self):
        """
        This is the main function where processing happens.
        It's called by PyG automatically if the 'processed' directory is empty
        or missing files from `processed_file_names`.
        """
        logging.info(f"Starting dataset processing...")
        
        # 1. Read the id_file
        with open(self.id_file, 'r') as f:
            pdb_ids_to_process = [line.strip() for line in f]
        
        logging.info(f"Found {len(pdb_ids_to_process)} PDB IDs to process.")
        
        num_processed = 0
        num_failed = 0
        
        # Iterate over all IDs and attempt to process them
        for pdb_id in tqdm(pdb_ids_to_process, desc="Processing Complexes"):
            
            # Define the path for the final processed file
            processed_path = Path(self.processed_dir) / f"{pdb_id}.pt"
            
            # This check allows us to resume processing if it was interrupted
            if processed_path.exists():
                # We should also check if the file is valid by loading it
                try:
                    # `weights_only=False` is crucial here
                    torch.load(processed_path, weights_only=False)
                    num_processed += 1
                    continue # The file is valid and exists, skip it
                except Exception:
                    logging.warning(f"Found corrupt file, deleting: {processed_path}")
                    processed_path.unlink() # Delete corrupt file and re-process
            
            # 2. Call our preprocessor from `preprocess.py`
            # This handles all the RDKit parsing logic
            complex_data = pp.process_complex(pdb_id, self.raw_dir)
            
            # If RDKit parsing fails, `process_complex` returns None
            if complex_data is None:
                logging.warning(f"Skipping {pdb_id}: preprocessing failed.")
                num_failed += 1
                continue
            
            try:
                # 3. Build the HeteroData object
                data = HeteroData()
                
                # --- Add Ligand Data (as per schema) ---
                ligand = complex_data['ligand']
                data['ligand'].x = ligand['x']
                data['ligand'].pos = ligand['pos']
                # Define the edge store for intramolecular ligand bonds
                data['ligand', 'intra', 'ligand'].edge_index = ligand['edge_index']
                
                # --- Add Receptor Data (as per schema) ---
                receptor = complex_data['receptor']
                data['receptor'].x = receptor['x']
                data['receptor'].pos = receptor['pos']
                
                # --- Dynamically Build Receptor Edges (as per schema) ---
                # Build k-NN graph on receptor C-Alpha positions
                receptor_edges = knn_graph(
                    receptor['pos'], 
                    k=self.k_receptor_neighbors, 
                    batch=None # This is a single graph
                )
                # Define the edge store for intramolecular receptor connections
                data['receptor', 'intra', 'receptor'].edge_index = receptor_edges
                
                # Add the PDB ID for easy reference and debugging
                data.pdb_id = pdb_id
                
                # Apply pre-filtering (if any)
                if self.pre_filter is not None and not self.pre_filter(data):
                    logging.warning(f"Skipping {pdb_id}: pre_filter returned False.")
                    num_failed += 1
                    continue

                # Apply pre-transformation (if any)
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                
                # 4. Save the HeteroData object to disk
                torch.save(data, processed_path)
                num_processed += 1

            except Exception as e:
                # Log the full error for debugging
                logging.error(f"Failed to build HeteroData for {pdb_id}: {e}", exc_info=True)
                num_failed += 1

        logging.info(f"Processing complete.")
        logging.info(f"Successfully processed (or found existing): {num_processed}")
        logging.info(f"Failed: {num_failed}")
        
        # Note: We do NOT scan for `self.processed_ids` here.
        # It is now done at the *end* of the __init__ method to ensure
        # the list is correct even if `process()` was not called.

if __name__ == '__main__':
    # A simple test script to run this file directly
    # This won't work correctly because of relative imports
    # Use the script in 'scripts/check_dataset.py' instead
    print("This file contains the PDBBindPyG Dataset class.")
    print("To test, run 'python scripts/check_dataset.py'")