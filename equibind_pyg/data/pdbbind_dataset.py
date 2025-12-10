"""
PDBBindPyG dataset.

This module defines a PyTorch Geometric `Dataset` for protein–ligand
complexes from the PDBBind benchmark. Each item is returned as a
`HeteroData` object with separate `ligand` and `receptor` node types:

- ligand: atom-level graph with 3D coordinates and 28-dim features
- receptor: residue-level graph (Cα) with 3D coordinates and 21-dim features

The dataset handles:

- Reading PDB IDs from an ID list file.
- Locating raw `.pdb` and `.sdf` files under a PDBBind-style layout.
- Calling `preprocess.process_complex` to parse and featurize complexes.
- Assembling `HeteroData` objects.
- Saving processed data to disk (`.pt` files) under `processed/`.
- Skipping and logging complexes that fail preprocessing.
- Reusing cached files unless `force_reprocess=True`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import torch
from torch_geometric.data import Dataset, HeteroData
from tqdm import tqdm
from torch_geometric.nn import knn_graph

from equibind_pyg.data import preprocess as pp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class PDBBindPyG(Dataset):
    """
    PyTorch Geometric dataset for PDBBind protein–ligand complexes.

    Parameters
    ----------
    root : str
        Root directory where processed data will be stored (e.g. `data/pyg_data`).
    id_file : str
        Path to a text file containing one PDB ID per line (e.g. `data/train_ids.txt`).
    k_receptor_neighbors : int, optional
        Number of nearest neighbors for the receptor k-NN graph, by default 30.
    force_reprocess : bool, optional
        If True, remove any existing `.pt` files under `processed/` and rebuild them.

    Raw layout (expected)
    ---------------------
    root.parent is expected to contain the raw PDBBind files in a `v2020/` folder:

        data/
          v2020/
            1a0q/
              1a0q_ligand.sdf
              1a0q_protein.pdb
            ...
          train_ids.txt
          val_ids.txt
          test_ids.txt
          pyg_data/
            processed/
              1a0q.pt
              10gs.pt
              ...

    This class does not handle downloading PDBBind. The user is expected to
    place the files in the layout described above.
    """

    def __init__(
        self,
        root: str,
        id_file: str,
        k_receptor_neighbors: int = 30,
        force_reprocess: bool = False,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ) -> None:
        self.id_file = id_file
        self.k_receptor_neighbors = k_receptor_neighbors
        self.raw_data_dir = Path(root).parent
        self.processed_ids: list[str] = []

        if force_reprocess and Path(self.processed_dir).exists():
            logging.warning(
                "force_reprocess=True. Deleting all files in %s",
                self.processed_dir,
            )
            deleted_count = 0
            for f in Path(self.processed_dir).glob("*.pt"):
                f.unlink()
                deleted_count += 1
            logging.warning("Deleted %d cached files.", deleted_count)

        super().__init__(root, transform, pre_transform, pre_filter)

        # Restrict to IDs that both appear in id_file and have a processed `.pt` file.
        with open(self.id_file, "r") as f:
            desired_ids = {line.strip() for line in f if line.strip()}

        all_files = Path(self.processed_dir).glob("*.pt")
        processed_ids_all = {
            f.stem
            for f in all_files
            if f.name not in ["pre_filter.pt", "pre_transform.pt"]
        }

        self.processed_ids = sorted(processed_ids_all & desired_ids)

        if not self.processed_ids:
            logging.warning(
                "PDBBindPyG: no processed complexes found for id_file=%s.",
                self.id_file,
            )
        else:
            logging.info(
                "PDBBindPyG: %d complexes found for split defined by %s.",
                len(self.processed_ids),
                self.id_file,
            )

    # -------------------------------------------------------------------------
    # Properties required by PyG Dataset
    # -------------------------------------------------------------------------

    @property
    def raw_dir(self) -> str:
        """Directory in which PyG expects the raw input files."""
        return str(self.raw_data_dir)

    @property
    def raw_file_names(self) -> List[str]:
        """
        A list of filenames that must exist in `raw_dir` for the dataset
        to be considered "downloaded" by PyG.

        Here we only require the ID file to be present.
        """
        id_path = Path(self.id_file)
        if not id_path.exists():
            raise FileNotFoundError(f"ID file not found: {self.id_file}")
        return [id_path.name]

    @property
    def processed_file_names(self) -> List[str]:
        """
        A list of expected processed filenames. PyG will call `process()` if
        any of these files are missing.
        """
        with open(self.id_file, "r") as f:
            pdb_ids = [line.strip() for line in f if line.strip()]
        return [f"{pdb_id}.pt" for pdb_id in pdb_ids]

    # -------------------------------------------------------------------------
    # Core Dataset methods
    # -------------------------------------------------------------------------

    def len(self) -> int:
        """Return the number of successfully processed complexes."""
        return len(self.processed_ids)

    def get(self, idx: int) -> HeteroData:
        """Load a single processed complex as `HeteroData`."""
        pdb_id = self.processed_ids[idx]
        file_path = Path(self.processed_dir) / f"{pdb_id}.pt"

        try:
            data = torch.load(file_path, weights_only=False)
        except TypeError:
            data = torch.load(file_path)

        data.pdb_id = pdb_id

        # Ensure ground-truth ligand coordinates are present for loss computation.
        if not hasattr(data, "ligand_pos_bound"):
            data.ligand_pos_bound = data["ligand"].pos.clone()

        return data

    def download(self) -> None:
        """
        No automatic download is implemented.

        PDBBind must be downloaded and unpacked by the user under
        `raw_data_dir / "v2020"`.
        """
        return

    def process(self) -> None:
        """
        Main processing loop.

        For each PDB ID in `id_file` this method:

          1. Calls `preprocess.process_complex()` to parse ligand + receptor.
          2. Builds a `HeteroData` object with:
             - ligand: x, pos, edge_index
             - receptor: x, pos, edge_index (k-NN graph)
          3. Saves the result to `processed_dir / f"{pdb_id}.pt"`.

        Complexes that fail preprocessing or fail any sanity checks are skipped
        and logged.
        """
        logging.info("Starting dataset processing...")

        with open(self.id_file, "r") as f:
            pdb_ids_to_process = [line.strip() for line in f if line.strip()]

        logging.info("Found %d PDB IDs to process.", len(pdb_ids_to_process))

        num_processed = 0
        num_failed = 0

        for pdb_id in tqdm(pdb_ids_to_process, desc="Processing complexes"):
            processed_path = Path(self.processed_dir) / f"{pdb_id}.pt"

            if processed_path.exists():
                try:
                    _ = torch.load(processed_path, weights_only=False)
                    num_processed += 1
                    continue
                except Exception:
                    logging.warning("Found corrupt file, deleting: %s", processed_path)
                    processed_path.unlink()

            complex_data = pp.process_complex(pdb_id, Path(self.raw_dir))
            if complex_data is None:
                logging.warning("Skipping %s: preprocessing failed.", pdb_id)
                num_failed += 1
                continue

            try:
                data = self._build_heterodata(pdb_id, complex_data)

                if self.pre_filter is not None and not self.pre_filter(data):
                    logging.warning(
                        "Skipping %s: pre_filter returned False.", pdb_id
                    )
                    num_failed += 1
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                torch.save(data, processed_path)
                num_processed += 1

            except Exception as e:
                logging.error(
                    "Failed to build HeteroData for %s: %s", pdb_id, repr(e),
                    exc_info=True,
                )
                num_failed += 1

        logging.info("Processing complete.")
        logging.info("Successfully processed (or found existing): %d", num_processed)
        logging.info("Failed: %d", num_failed)

    # -------------------------------------------------------------------------
    # Internal helper
    # -------------------------------------------------------------------------

    def _build_heterodata(self, pdb_id: str, complex_data: dict) -> HeteroData:
        """
        Build a `HeteroData` object from the nested dict returned by
        `preprocess.process_complex`.

        The input `complex_data` is expected to be:

            {
                'ligand': {
                    'x': FloatTensor[N_l, 28],
                    'pos': FloatTensor[N_l, 3],
                    'edge_index': LongTensor[2, E_l],
                    ...
                },
                'receptor': {
                    'x': FloatTensor[N_r, 21],
                    'pos': FloatTensor[N_r, 3],
                    ...
                }
            }
        """
        ligand = complex_data["ligand"]
        receptor = complex_data["receptor"]

        data = HeteroData()

        # Ligand
        x_lig = ligand["x"]
        pos_lig = ligand["pos"]
        edge_lig = ligand["edge_index"]

        assert x_lig.size(0) == pos_lig.size(0), "Ligand x and pos must have same number of nodes"

        data["ligand"].x = x_lig
        data["ligand"].pos = pos_lig
        data["ligand", "intra", "ligand"].edge_index = edge_lig

        # Receptor
        x_rec = receptor["x"]
        pos_rec = receptor["pos"]

        assert x_rec.size(0) == pos_rec.size(0), "Receptor x and pos must have same number of nodes"

        data["receptor"].x = x_rec
        data["receptor"].pos = pos_rec

        # Build k-NN graph on receptor positions
        receptor_edges = knn_graph(
            pos_rec,
            k=self.k_receptor_neighbors,
            batch=None,
        )
        data["receptor", "intra", "receptor"].edge_index = receptor_edges

        data.pdb_id = pdb_id
        return data
