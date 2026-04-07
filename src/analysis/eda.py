"""
This is a  basic exploratory data analysis for HyMARS hyperspectral imaging dataset
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import ticker
from scipy.stats import entropy, skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path
import warnings

from utils.logger import LoggerSingleton
from utils.load_data import HyMarsDataModule

warnings.filterwarnings('ignore')

logger = LoggerSingleton().logger

plt.rcParams['figure.dpi'] = 300 
plt.rcParams['savefig.dpi'] = 300 * 3
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['figure.constrained_layout.use'] = False

class HyMarsEDA:
    """EDA class for HyMars hyperspectral dataset."""
    
    def __init__(self, data_dir: str, output_dir: str):
        """
        Initialize EDA analyzer.
        
        Parameters
        ----------
        data_dir : str
            Path to directory containing .mat files
        output_dir : str
            Path to directory for saving plots
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) / 'eda'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_module = None
        self.datasets = {}
        self.groundtruths = {}
        self.metadata = None
        
    def load_data(self):
        """Load all datasets using HyMarsDataModule (PyTorch compatible)."""
        logger.info("Loading hyperspectral data via DataModule...")
        
        self.data_module = HyMarsDataModule(
            data_dir=str(self.data_dir),
            batch_size=32,
            patch_size=1,
            normalize=False,
            num_workers=0
        )
        
        self.metadata = self.data_module.metadata
        
        for key in self.metadata.keys():
            try:
                self.datasets[key] = self.data_module.get_raw_data(key)
                logger.info(f"Loaded {key}: {self.datasets[key].shape}")
            except Exception as e:
                logger.error(f"Error loading {key}: {e}")
            
            try:
                gt = self.data_module.get_ground_truth(key)
                if gt is not None:
                    self.groundtruths[key] = gt
                    logger.info(f"Loaded {key} ground truth: {gt.shape}")
            except Exception as e:
                logger.error(f"Error loading {key} GT: {e}")
    
    def compute_dataset_statistics(self):
        """Compute comprehensive statistics for each dataset."""
        stats = {}
        
        for key, data in self.datasets.items():
            pixels, height, width = data.shape[0], data.shape[1], data.shape[2]
            data_reshaped = data.reshape(-1, data.shape[2])
            
            stats[key] = {
                'shape': data.shape,
                'n_pixels': pixels * height,
                'n_bands': data.shape[2],
                'mean': np.mean(data_reshaped, axis=0),
                'std': np.std(data_reshaped, axis=0),
                'min': np.min(data_reshaped, axis=0),
                'max': np.max(data_reshaped, axis=0),
                'median': np.median(data_reshaped, axis=0),
                'global_mean': np.mean(data_reshaped),
                'global_std': np.std(data_reshaped),
                'global_min': np.min(data_reshaped),
                'global_max': np.max(data_reshaped),
                'skewness': skew(data_reshaped, axis=0),
                'kurtosis': kurtosis(data_reshaped, axis=0),
                'entropy': np.array([entropy(np.abs(data_reshaped[:, i])) 
                                     for i in range(data_reshaped.shape[1])]),
                'snr': self._compute_snr(data_reshaped),
                'dynamic_range': np.max(data_reshaped, axis=0) - np.min(data_reshaped, axis=0)
            }
        
        return stats
    
    def _compute_snr(self, data):
        """Compute Signal-to-Noise Ratio for each band."""
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)

        snr = np.divide(mean, std, where=std>0, out=np.zeros_like(mean))
        return snr
    
    def plot_overview_table(self, stats):
        """Create a publication-quality overview table."""
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.axis('tight')
        ax.axis('off')
        
        table_data = []
        headers = ['Dataset', 'Dimensions (H×W)', 'Spectral Bands', 
                   'Pixels', 'Mean Intensity', 'Std Dev', 'Dynamic Range']
        
        for key in sorted(self.datasets.keys()):
            info = self.metadata[key]
            s = stats[key]
            table_data.append([
                info['name'],
                f"{s['shape'][1]} × {s['shape'][2]}",
                f"{s['n_bands']}",
                f"{s['n_pixels']:,}",
                f"{s['global_mean']:.4f}",
                f"{s['global_std']:.4f}",
                f"{s['dynamic_range'].mean():.4f}"
            ])
        
        table = ax.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center',
                        colWidths=[0.18, 0.16, 0.12, 0.12, 0.14, 0.14, 0.14])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
                else:
                    table[(i, j)].set_facecolor('white')
        
        plt.title('HyMars Dataset Overview Statistics', 
                 fontsize=13, fontweight='bold', pad=20)
        
        plt.savefig(self.output_dir / '01_dataset_overview.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()
        logger.info("Saved: 01_dataset_overview.png")
    
    def plot_spatial_dimensions(self):
        """Visualize spatial dimensions of datasets."""
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for ax, (key, color) in zip(axes, zip(sorted(self.datasets.keys()), colors)):
            shape = self.datasets[key].shape
            h, w = shape[1], shape[2]
            
            ax.barh(['Height', 'Width'], [h, w], color=color, alpha=0.7, edgecolor='black', linewidth=1.5)
            ax.set_xlabel('Pixels', fontsize=11, fontweight='bold')
            ax.set_title(f"{self.metadata[key]['name']}\n({h} × {w} px)", 
                        fontsize=11, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            for i, v in enumerate([h, w]):
                ax.text(v + 10, i, f'{v}', va='center', fontweight='bold')
        
        plt.suptitle('Spatial Dimensions of Hyperspectral Images', 
                    fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / '02_spatial_dimensions.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()
        logger.info("Saved: 02_spatial_dimensions.png")
    
    def plot_spectral_band_statistics(self, stats):
        """Plot comprehensive spectral statistics across bands."""
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()
        
        metrics = [
            ('mean', 'Mean Spectral Intensity', 0),
            ('std', 'Standard Deviation', 1),
            ('max', 'Maximum Reflectance', 2),
            ('dynamic_range', 'Dynamic Range', 3),
            ('skewness', 'Skewness (Distribution Asymmetry)', 4),
            ('snr', 'Signal-to-Noise Ratio', 5)
        ]
        
        colors_datasets = {'Holden': '#FF6B6B', 'NiliFossae': '#4ECDC4', 'Utopia': '#45B7D1'}
        
        for ax, (metric, title, idx) in zip(axes, metrics):
            for key in sorted(self.datasets.keys()):
                if metric in stats[key]:
                    ax.plot(stats[key][metric], label=self.metadata[key]['name'], 
                           linewidth=2.2, color=colors_datasets[key], alpha=0.8)
            
            ax.set_xlabel('Spectral Band Index', fontsize=10, fontweight='bold')
            ax.set_ylabel(title, fontsize=10, fontweight='bold')
            ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='best', framealpha=0.95)
        
        plt.suptitle('Comprehensive Spectral Band Analysis', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(self.output_dir / '03_spectral_statistics.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()
        logger.info("Saved: 03_spectral_statistics.png")
    
    def plot_intensity_distributions(self, stats):
        """Plot intensity distributions for each dataset."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for ax, (key, color) in zip(axes, zip(sorted(self.datasets.keys()), colors)):
            data_reshaped = self.datasets[key].reshape(-1, self.datasets[key].shape[2])
            all_values = data_reshaped.flatten()
            
            ax.hist(all_values, bins=100, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
            ax.axvline(stats[key]['global_mean'], color='red', linestyle='--', 
                      linewidth=2.5, label=f"Mean: {stats[key]['global_mean']:.4f}")
            ax.axvline(np.median(all_values), color='green', linestyle='--', 
                      linewidth=2.5, label=f"Median: {np.median(all_values):.4f}")
            
            ax.set_xlabel('Pixel Intensity', fontsize=11, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
            ax.set_title(f"{self.metadata[key]['name']}\n{len(all_values):,} pixels", 
                        fontsize=11, fontweight='bold')
            ax.legend(loc='upper right', framealpha=0.95)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_yscale('log')
        
        plt.suptitle('Intensity Distribution Analysis', 
                    fontsize=13, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(self.output_dir / '04_intensity_distributions.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()
        logger.info("Saved: 04_intensity_distributions.png")
    
    def plot_band_correlation_heatmap(self):
        """Plot correlation heatmaps for spectral bands (downsampled for clarity)."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for ax, key in zip(axes, sorted(self.datasets.keys())):
            data_reshaped = self.datasets[key].reshape(-1, self.datasets[key].shape[2])
            
            n_bands = data_reshaped.shape[1]
            step = max(1, n_bands // 50)
            band_indices = np.arange(0, n_bands, step)
            
            data_subset = data_reshaped[:, band_indices]
            corr_matrix = np.corrcoef(data_subset.T)
            
            im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            ax.set_xlabel('Band Index (subsampled)', fontsize=10, fontweight='bold')
            ax.set_ylabel('Band Index (subsampled)', fontsize=10, fontweight='bold')
            ax.set_title(f"{self.metadata[key]['name']}\n{len(band_indices)} bands shown", 
                        fontsize=11, fontweight='bold')
            
            cbar = plt.colorbar(im, ax=ax, label='Correlation')
        
        plt.suptitle('Spectral Band Correlation Analysis (Subsampled)', 
                    fontsize=13, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(self.output_dir / '05_band_correlation_heatmap.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()
        logger.info("Saved: 05_band_correlation_heatmap.png")
    
    def plot_pca_analysis(self):
        """Perform and visualize PCA analysis."""
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        colors_list = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for ax, (key, color) in zip(axes[0, :], 
                                    zip(sorted(self.datasets.keys()), colors_list)):
            data_reshaped = self.datasets[key].reshape(-1, self.datasets[key].shape[2])
            
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_reshaped)
            
            pca = PCA()
            pca.fit(data_scaled)
            
            cumsum = np.cumsum(pca.explained_variance_ratio_)
            ax.plot(cumsum, linewidth=2.5, color=color, marker='o', 
                   markersize=3, markevery=max(1, len(cumsum)//20))
            ax.axhline(0.95, color='red', linestyle='--', linewidth=2, 
                      label='95% threshold', alpha=0.7)
            n_components_95 = np.argmax(cumsum >= 0.95) + 1
            ax.axvline(n_components_95, color='green', linestyle='--', 
                      linewidth=2, label=f'{n_components_95} components', alpha=0.7)
            
            ax.set_xlabel('Principal Component', fontsize=10, fontweight='bold')
            ax.set_ylabel('Cumulative Variance Explained', fontsize=10, fontweight='bold')
            ax.set_title(f"{self.metadata[key]['name']}", fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='lower right', framealpha=0.95)
            ax.set_ylim([0, 1.05])
        
        for ax, (key, color) in zip(axes[1, :], 
                                    zip(sorted(self.datasets.keys()), colors_list)):
            data_reshaped = self.datasets[key].reshape(-1, self.datasets[key].shape[2])
            
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_reshaped)
            
            pca = PCA(n_components=3)
            pc_data = pca.fit_transform(data_scaled)
            
            scatter = ax.scatter(pc_data[:, 0], pc_data[:, 1], 
                               c=pc_data[:, 2], cmap='viridis', 
                               s=20, alpha=0.6, edgecolors='none')
            
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', 
                         fontsize=10, fontweight='bold')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', 
                         fontsize=10, fontweight='bold')
            ax.set_title(f"{self.metadata[key]['name']} (colored by PC3)", 
                        fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})', 
                          fontsize=9, fontweight='bold')
        
        plt.suptitle('Principal Component Analysis (PCA)', 
                    fontsize=13, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(self.output_dir / '06_pca_analysis.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()
        logger.info("Saved: 06_pca_analysis.png")
    
    def plot_rgb_composites(self):
        """Create synthetic RGB composites from principal components."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for ax, key in zip(axes, sorted(self.datasets.keys())):
            data = self.datasets[key]
            height, width, n_bands = data.shape
            
            band_indices = [
                int(n_bands * 0.1),
                int(n_bands * 0.5),
                int(n_bands * 0.9)
            ]
            
            rgb = np.stack([
                data[:, :, band_indices[0]],
                data[:, :, band_indices[1]],
                data[:, :, band_indices[2]]
            ], axis=-1)
            
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
            rgb = np.clip(rgb, 0, 1)
            
            rgb = np.power(rgb, 0.6)
            
            ax.imshow(rgb)
            ax.set_title(f"{self.metadata[key]['name']}\n"
                        f"Bands {band_indices[0]}, {band_indices[1]}, {band_indices[2]}",
                        fontsize=11, fontweight='bold')
            ax.axis('off')
        
        plt.suptitle('Synthetic RGB Composites (Band Combination)', 
                    fontsize=13, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(self.output_dir / '07_rgb_composites.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()
        logger.info("Saved: 07_rgb_composites.png")
    
    def plot_ground_truth_analysis(self):
        """Analyze and visualize ground truth labels."""
        if not self.groundtruths:
            logger.warning("No ground truth data available, skipping GT analysis")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        for ax, key in zip(axes[0, :], sorted(self.groundtruths.keys())):
            gt = self.groundtruths[key].squeeze()
            
            im = ax.imshow(gt, cmap='tab20', interpolation='nearest')
            ax.set_title(f"{self.metadata[key]['name']}", fontsize=11, fontweight='bold')
            ax.axis('off')
            
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Class Label', fontsize=9, fontweight='bold')
        
        for ax, key in zip(axes[1, :], sorted(self.groundtruths.keys())):
            gt = self.groundtruths[key].squeeze()
            unique_labels, counts = np.unique(gt, return_counts=True)
            
            mask = unique_labels != 0
            unique_labels = unique_labels[mask]
            counts = counts[mask]
            
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
            bars = ax.bar(unique_labels, counts, color=colors, edgecolor='black', linewidth=1)
            
            ax.set_xlabel('Mineral Class Label', fontsize=10, fontweight='bold')
            ax.set_ylabel('Pixel Count', fontsize=10, fontweight='bold')
            ax.set_title(f"{self.metadata[key]['name']} Label Distribution", 
                        fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=8)
        
        plt.suptitle('Ground Truth Label Analysis', 
                    fontsize=13, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(self.output_dir / '08_ground_truth_analysis.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()
        logger.info("Saved: 08_ground_truth_analysis.png")
    
    def plot_spectral_signature_analysis(self, stats):
        """Analyze representative spectral signatures."""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        colors_ds = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for ax, (key, color) in zip(axes, zip(sorted(self.datasets.keys()), colors_ds)):
            data_reshaped = self.datasets[key].reshape(-1, self.datasets[key].shape[2])
            
            mean_spectrum = stats[key]['mean']
            std_spectrum = stats[key]['std']
            x = np.arange(len(mean_spectrum))
            
            ax.plot(x, mean_spectrum, color=color, linewidth=2.5, label='Mean', zorder=3)
            ax.fill_between(x, mean_spectrum - std_spectrum, mean_spectrum + std_spectrum, 
                           color=color, alpha=0.25, label='±1 Std Dev', zorder=1)
            
            ax.fill_between(x, stats[key]['min'], stats[key]['max'], 
                           color=color, alpha=0.1, label='Min-Max Range', zorder=0)
            
            ax.set_xlabel('Spectral Band Index', fontsize=11, fontweight='bold')
            ax.set_ylabel('Reflectance Intensity', fontsize=11, fontweight='bold')
            ax.set_title(f"{self.metadata[key]['name']}\n{data_reshaped.shape[0]:,} pixels analyzed", 
                        fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='best', framealpha=0.95)
        
        plt.suptitle('Mean Spectral Signatures with Variability', 
                    fontsize=13, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(self.output_dir / '09_spectral_signatures.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()
        logger.info("Saved: 09_spectral_signatures.png")
    
    def plot_band_quality_assessment(self, stats):
        """Assess quality metrics for spectral bands."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        colors_ds = {'Holden': '#FF6B6B', 'NiliFossae': '#4ECDC4', 'Utopia': '#45B7D1'}
        
        ax = axes[0]
        for key in sorted(self.datasets.keys()):
            cv = stats[key]['std'] / (stats[key]['mean'] + 1e-8)
            consistency = 1.0 / (1.0 + cv)  # Transform to [0, 1]
            ax.plot(consistency, label=self.metadata[key]['name'], 
                   linewidth=2.2, color=colors_ds[key])
        ax.set_xlabel('Spectral Band Index', fontsize=11, fontweight='bold')
        ax.set_ylabel('Signal Consistency', fontsize=11, fontweight='bold')
        ax.set_title('Band Signal Consistency', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', framealpha=0.95)
        ax.set_ylim([0, 1])
        
        ax = axes[1]
        for key in sorted(self.datasets.keys()):
            ax.plot(stats[key]['entropy'], label=self.metadata[key]['name'], 
                   linewidth=2.2, color=colors_ds[key])
        ax.set_xlabel('Spectral Band Index', fontsize=11, fontweight='bold')
        ax.set_ylabel('Shannon Entropy', fontsize=11, fontweight='bold')
        ax.set_title('Spectral Information Content', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', framealpha=0.95)
        
        ax = axes[2]
        for key in sorted(self.datasets.keys()):
            ax.plot(stats[key]['kurtosis'], label=self.metadata[key]['name'], 
                   linewidth=2.2, color=colors_ds[key])
        ax.axhline(0, color='black', linestyle=':', linewidth=1, alpha=0.5)
        ax.set_xlabel('Spectral Band Index', fontsize=11, fontweight='bold')
        ax.set_ylabel('Kurtosis', fontsize=11, fontweight='bold')
        ax.set_title('Spectral Distribution Kurtosis', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', framealpha=0.95)
        
        ax = axes[3]
        for key in sorted(self.datasets.keys()):
            cv = stats[key]['std'] / (stats[key]['mean'] + 1e-8)
            ax.plot(cv, label=self.metadata[key]['name'], 
                   linewidth=2.2, color=colors_ds[key])
        ax.set_xlabel('Spectral Band Index', fontsize=11, fontweight='bold')
        ax.set_ylabel('Coefficient of Variation', fontsize=11, fontweight='bold')
        ax.set_title('Relative Spectral Variability', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', framealpha=0.95)
        
        plt.suptitle('Comprehensive Band Quality Assessment', 
                    fontsize=13, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(self.output_dir / '10_band_quality_assessment.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()
        logger.info("Saved: 10_band_quality_assessment.png")
    
    def plot_spatial_statistics(self):
        """Analyze spatial statistics (mean image, variance, etc)."""
        fig, axes = plt.subplots(3, 3, figsize=(14, 12))
        
        for col, key in enumerate(sorted(self.datasets.keys())):
            data = self.datasets[key]
            
            mean_image = np.mean(data, axis=2)
            
            var_image = np.var(data, axis=2)
            
            std_image = np.std(data, axis=2)
            
            ax = axes[0, col]
            im = ax.imshow(mean_image, cmap='RdYlBu_r')
            ax.set_title(f"{self.metadata[key]['name']}\nMean Spectrum", 
                        fontsize=11, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            ax = axes[1, col]
            im = ax.imshow(var_image, cmap='hot')
            ax.set_title(f"Spectral Variance", fontsize=11, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            ax = axes[2, col]
            im = ax.imshow(std_image, cmap='viridis')
            ax.set_title(f"Spectral Std Dev", fontsize=11, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.suptitle('Spatial Statistical Maps', 
                    fontsize=13, fontweight='bold', y=0.998)
        plt.tight_layout()
        plt.savefig(self.output_dir / '11_spatial_statistics.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()
        logger.info("Saved: 11_spatial_statistics.png")
    
    def plot_dimensionality_comparison(self):
        """Compare PCA dimensionality reduction results across datasets."""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        colors_list = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for col, (key, color) in enumerate(zip(sorted(self.datasets.keys()), colors_list)):
            data_reshaped = self.datasets[key].reshape(-1, self.datasets[key].shape[2])
            
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_reshaped)
            
            n_samples = min(2000, data_scaled.shape[0])
            indices = np.random.choice(data_scaled.shape[0], n_samples, replace=False)
            data_sample = data_scaled[indices, :]
            
            pca = PCA(n_components=2)
            pca_data = pca.fit_transform(data_sample)
            
            ax = axes[col]
            scatter = ax.scatter(pca_data[:, 0], pca_data[:, 1], 
                               c=np.arange(len(pca_data)), cmap='viridis', 
                               s=30, alpha=0.6, edgecolors='none')
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', 
                         fontsize=11, fontweight='bold')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', 
                         fontsize=11, fontweight='bold')
            ax.set_title(f"{self.metadata[key]['name']}\nPCA (2D, {n_samples:,} pixels)", 
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Pixel Index', fontsize=9)
        
        plt.suptitle('Principal Component Analysis (2D Projection)', 
                    fontsize=13, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(self.output_dir / '12_dimensionality_reduction.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()
        logger.info("Saved: 12_dimensionality_reduction.png")
    
    def plot_missing_data_analysis(self):
        """Analyze missing or problematic data."""
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        
        for ax, key in zip(axes, sorted(self.datasets.keys())):
            data_reshaped = self.datasets[key].reshape(-1, self.datasets[key].shape[2])
            
            nan_per_band = np.sum(np.isnan(data_reshaped), axis=0)
            inf_per_band = np.sum(np.isinf(data_reshaped), axis=0)
            zero_per_band = np.sum(data_reshaped == 0, axis=0)
            
            x = np.arange(len(nan_per_band))
            
            ax.plot(nan_per_band, label='NaN values', linewidth=2, color='red')
            ax.plot(inf_per_band, label='Inf values', linewidth=2, color='orange')
            ax.plot(zero_per_band / 100, label='Zero values (÷100)', linewidth=2, color='blue')
            
            ax.set_xlabel('Spectral Band Index', fontsize=10, fontweight='bold')
            ax.set_ylabel('Count', fontsize=10, fontweight='bold')
            ax.set_title(f"{self.metadata[key]['name']}", fontsize=11, fontweight='bold')
            ax.legend(loc='best', framealpha=0.95)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Missing/Problematic Data Detection', 
                    fontsize=13, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(self.output_dir / '13_missing_data_analysis.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()
        logger.info("Saved: 13_missing_data_analysis.png")
    
    def plot_statistical_summary(self, stats):
        """Create a comprehensive statistical summary visualization."""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
        
        keys = sorted(self.datasets.keys())
        
        for idx, key in enumerate(keys):
            ax = fig.add_subplot(gs[0, idx])
            metrics = ['Mean', 'Std', 'Min', 'Max']
            values = [
                stats[key]['global_mean'],
                stats[key]['global_std'],
                stats[key]['global_min'],
                stats[key]['global_max']
            ]
            colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
            bars = ax.bar(metrics, values, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
            ax.set_ylabel('Value', fontsize=10, fontweight='bold')
            ax.set_title(f"{self.metadata[key]['name']}\nGlobal Statistics", 
                        fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax = fig.add_subplot(gs[1, idx])
            data_reshaped = self.datasets[key].reshape(-1, self.datasets[key].shape[2])
            
            n_bands = data_reshaped.shape[1]
            n_groups = min(10, n_bands)
            group_size = n_bands // n_groups
            
            boxplot_data = []
            for i in range(n_groups):
                start = i * group_size
                end = start + group_size if i < n_groups - 1 else n_bands
                boxplot_data.append(data_reshaped[:, start:end].flatten())
            
            bp = ax.boxplot(boxplot_data, labels=[f'G{i+1}' for i in range(n_groups)],
                          patch_artist=True, widths=0.6)
            
            for patch in bp['boxes']:
                patch.set_facecolor('#3498db')
                patch.set_alpha(0.7)
            
            ax.set_ylabel('Intensity', fontsize=10, fontweight='bold')
            ax.set_xlabel('Spectral Band Groups', fontsize=10, fontweight='bold')
            ax.set_title('Intensity Distribution by Band Group', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            ax = fig.add_subplot(gs[2, idx])
            quantiles = np.linspace(0, 1, 101)
            quantile_values = np.quantile(data_reshaped.flatten(), quantiles)
            
            ax.plot(quantiles * 100, quantile_values, linewidth=2.5, color='#9b59b6', marker='o',
                   markersize=3, markevery=10)
            ax.axvline(50, color='red', linestyle='--', linewidth=1.5, label='Median', alpha=0.7)
            ax.axvline(25, color='blue', linestyle='--', linewidth=1.5, label='Q1/Q3', alpha=0.7)
            ax.axvline(75, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
            
            ax.set_xlabel('Percentile (%)', fontsize=10, fontweight='bold')
            ax.set_ylabel('Intensity', fontsize=10, fontweight='bold')
            ax.set_title('Quantile Distribution', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=9)
        
        plt.suptitle('Comprehensive Statistical Summary', 
                    fontsize=14, fontweight='bold', y=0.998)
        plt.savefig(self.output_dir / '14_statistical_summary.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()
        logger.info("Saved: 14_statistical_summary.png")
    
    def plot_inter_dataset_comparison(self, stats):
        """Compare characteristics across datasets."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        keys = sorted(self.datasets.keys())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        x_pos = np.arange(len(keys))
        
        ax = axes[0]
        means = [stats[k]['global_mean'] for k in keys]
        stds = [stats[k]['global_std'] for k in keys]
        
        width = 0.35
        bars1 = ax.bar(x_pos - width/2, means, width, label='Mean', 
                       color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x_pos + width/2, stds, width, label='Std Dev', 
                       color=colors, alpha=0.5, edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel('Value', fontsize=11, fontweight='bold')
        ax.set_title('Global Intensity Statistics', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([self.metadata[k]['name'].split('(')[1].replace(')', '') for k in keys])
        ax.legend(framealpha=0.95)
        ax.grid(True, alpha=0.3, axis='y')
        
        ax = axes[1]
        n_bands = [stats[k]['n_bands'] for k in keys]
        n_pixels = [stats[k]['n_pixels'] for k in keys]
        
        bars1 = ax.bar(x_pos - width/2, n_bands, width, label='Spectral Bands', 
                       color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax2 = ax.twinx()
        bars2 = ax2.bar(x_pos + width/2, [p/1000 for p in n_pixels], width, 
                        label='Pixels (thousands)', color=colors, alpha=0.5, 
                        edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel('Spectral Bands', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Pixels (thousands)', fontsize=11, fontweight='bold')
        ax.set_title('Data Dimensionality', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([self.metadata[k]['name'].split('(')[1].replace(')', '') for k in keys])
        ax.grid(True, alpha=0.3, axis='y')
        
        ax = axes[2]
        dyn_ranges = [np.mean(stats[k]['dynamic_range']) for k in keys]
        
        bars = ax.bar(x_pos, dyn_ranges, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Average Dynamic Range', fontsize=11, fontweight='bold')
        ax.set_title('Spectral Dynamic Range', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([self.metadata[k]['name'].split('(')[1].replace(')', '') for k in keys])
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax = axes[3]
        
        snr_mean = [np.mean(stats[k]['snr']) for k in keys]
        entropy_mean = [np.mean(stats[k]['entropy']) for k in keys]
        
        bars1 = ax.bar(x_pos - width/2, snr_mean, width, label='Mean SNR', 
                       color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax2 = ax.twinx()
        bars2 = ax2.bar(x_pos + width/2, entropy_mean, width, label='Mean Entropy', 
                        color=colors, alpha=0.5, edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel('Signal-to-Noise Ratio', fontsize=11, fontweight='bold', color='black')
        ax2.set_ylabel('Shannon Entropy', fontsize=11, fontweight='bold', color='black')
        ax.set_title('Data Quality Metrics', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([self.metadata[k]['name'].split('(')[1].replace(')', '') for k in keys])
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Inter-Dataset Comparative Analysis', 
                    fontsize=13, fontweight='bold', y=0.998)
        plt.tight_layout()
        plt.savefig(self.output_dir / '15_inter_dataset_comparison.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()
        logger.info("Saved: 15_inter_dataset_comparison.png")
    
    def run_complete_analysis(self):
        """Execute the complete EDA pipeline."""
        logger.info("Starting comprehensive exploratory data analysis...")
        
        self.load_data()
        
        if not self.datasets:
            logger.error("No data files found. Aborting analysis.")
            return
        
        logger.info("Computing comprehensive statistics...")
        stats = self.compute_dataset_statistics()
        
        logger.info("Generating publication-quality visualizations...")
        
        self.plot_overview_table(stats)
        self.plot_spatial_dimensions()
        self.plot_spectral_band_statistics(stats)
        self.plot_intensity_distributions(stats)
        self.plot_band_correlation_heatmap()
        self.plot_pca_analysis()
        self.plot_rgb_composites()
        self.plot_ground_truth_analysis()
        self.plot_spectral_signature_analysis(stats)
        self.plot_band_quality_assessment(stats)
        self.plot_spatial_statistics()
        self.plot_dimensionality_comparison()
        self.plot_missing_data_analysis()
        self.plot_statistical_summary(stats)
        self.plot_inter_dataset_comparison(stats)
        
        logger.info(f"EDA complete. All visualizations saved to: {self.output_dir}")


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data"
    output_dir = project_root / "output"
    
    eda = HyMarsEDA(str(data_dir), str(output_dir))
    eda.run_complete_analysis()
