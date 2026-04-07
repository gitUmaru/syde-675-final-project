import numpy as np
from pathlib import Path
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset, DataLoader


class HyMarsDataset(Dataset):
    """PyTorch Dataset for hyperspectral image patches."""
    
    def __init__(self, data, patch_size=1, normalize=False):
        """
        Initialize hyperspectral dataset.
        
        Parameters
        ----------
        data : np.ndarray
            Hyperspectral image of shape (height, width, bands)
        patch_size : int
            Patch size for extraction
        normalize : bool
            Whether to normalize data
        """
        self.data = data
        self.patch_size = patch_size
        self.height, self.width, self.bands = data.shape
        self.normalize = normalize
        
        if normalize:
            self.data = self._normalize()
    
    def _normalize(self):
        """Normalize data to [0, 1] range."""
        vmin = self.data.min()
        vmax = self.data.max()
        return (self.data - vmin) / (vmax - vmin + 1e-8)
    
    def __len__(self):
        return self.height * self.width
    
    def __getitem__(self, idx):
        h = idx // self.width
        w = idx % self.width
        
        h_start = max(0, h - self.patch_size // 2)
        h_end = min(self.height, h + self.patch_size // 2 + 1)
        w_start = max(0, w - self.patch_size // 2)
        w_end = min(self.width, w + self.patch_size // 2 + 1)
        
        patch = self.data[h_start:h_end, w_start:w_end, :]
        patch = torch.from_numpy(patch).permute(2, 0, 1).float()
        
        return patch


class HyMarsDataModule:
    """Data module for HyMars hyperspectral datasets."""
    
    def __init__(self, data_dir, batch_size=32, patch_size=1, normalize=False, num_workers=0):
        """
        Initialize data module.
        
        Parameters
        ----------
        data_dir : str
            Path to directory containing .mat files
        batch_size : int
            Batch size for DataLoader
        patch_size : int
            Patch size for extraction
        normalize : bool
            Whether to normalize data
        num_workers : int
            Number of workers for DataLoader
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.normalize = normalize
        self.num_workers = num_workers
        
        self._raw_data = {}
        self._ground_truth = {}
        self._metadata = {}
        self._load_all_datasets()
    
    def _load_all_datasets(self):
        """Load all .mat files from data directory."""
        mat_files = sorted(self.data_dir.glob('*.mat'))
        
        dataset_names = set()
        for file in mat_files:
            name = file.stem.replace('_gt', '')
            dataset_names.add(name)
        
        for name in sorted(dataset_names):
            self._load_dataset(name)
    
    def _load_dataset(self, name):
        """Load a single dataset and its ground truth."""
        data_file = self.data_dir / f'{name}.mat'
        gt_file = self.data_dir / f'{name}_gt.mat'
        
        if data_file.exists():
            data = self._load_mat_file(data_file)
            self._raw_data[name] = data
            self._metadata[name] = {
                'name': name,
                'shape': data.shape,
                'dtype': str(data.dtype)
            }
        
        if gt_file.exists():
            gt = self._load_mat_file(gt_file)
            self._ground_truth[name] = gt
    
    def _load_mat_file(self, filepath):
        """Load .mat file and extract array."""
        mat = loadmat(filepath)
        
        for key, val in mat.items():
            if not key.startswith('__'):
                if isinstance(val, np.ndarray) and val.size > 0:
                    return val
        
        raise ValueError(f"No valid data found in {filepath}")
    
    @property
    def metadata(self):
        """Get metadata for all datasets."""
        return self._metadata
    
    def get_raw_data(self, name):
        """
        Get raw hyperspectral data for dataset.
        
        Parameters
        ----------
        name : str
            Dataset name
        
        Returns
        -------
        np.ndarray
            Raw data
        """
        if name not in self._raw_data:
            raise KeyError(f"Dataset {name} not found")
        return self._raw_data[name]
    
    def get_ground_truth(self, name):
        """
        Get ground truth labels for dataset.
        
        Parameters
        ----------
        name : str
            Dataset name
        
        Returns
        -------
        np.ndarray or None
            Ground truth data if available
        """
        return self._ground_truth.get(name, None)
    
    def get_dataloader(self, name, shuffle=False):
        """
        Get PyTorch DataLoader for dataset.
        
        Parameters
        ----------
        name : str
            Dataset name
        shuffle : bool
            Whether to shuffle data
        
        Returns
        -------
        DataLoader
            PyTorch DataLoader
        """
        data = self.get_raw_data(name)
        dataset = HyMarsDataset(data, self.patch_size, self.normalize)
        return DataLoader(dataset, batch_size=self.batch_size, 
                         shuffle=shuffle, num_workers=self.num_workers)
