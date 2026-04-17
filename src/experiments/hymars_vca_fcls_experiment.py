from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from .base_experiment import BaseExperiment
from ..models.vca import FCLS
from utils.load_data import HyMarsDataModule
from utils.logger import LoggerSingleton


plt.rcParams["figure.dpi"] = 200
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.linestyle"] = "--"


class HyMarsVCAFCLSExperiment(BaseExperiment):
    def __init__(self, config):
        super().__init__(config)
        self.logger = LoggerSingleton().logger
        self.data_module = None
        self.results = {}
        self.output_dir = Path(self.config.get("output_dir", "output/plots/vca_fcls"))
        self.data_dir = Path(self.config.get("data_dir", "data"))
        self.dataset_names = self.config.get("dataset_names")
        self.random_state = self.config.get("random_state", 0)
        self.vca_sample_size = int(self.config.get("vca_sample_size", 50000))
        self.n_endmembers = self.config.get("n_endmembers")
        self.fcls_max_iter = int(self.config.get("fcls_max_iter", 25))
        self.fcls_tol = float(self.config.get("fcls_tol", 1e-5))
        self.fcls_batch_size = int(self.config.get("fcls_batch_size", 8192))

    def setup(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_module = HyMarsDataModule(
            data_dir=str(self.data_dir),
            batch_size=32,
            patch_size=1,
            normalize=False,
            num_workers=0,
        )
        if self.dataset_names is None:
            self.dataset_names = sorted(self.data_module.metadata.keys())
        self.logger.info(f"Configured datasets: {', '.join(self.dataset_names)}")

    def _prepare_dataset(self, name):
        raw = self.data_module.get_raw_data(name)
        gt = self.data_module.get_ground_truth(name)
        x = raw.reshape(-1, raw.shape[-1]).astype(np.float32, copy=False)
        labels = None if gt is None else gt.reshape(-1).astype(np.int32, copy=False)
        return raw, x, labels

    def _fit_single_dataset(self, name):
        raw, x, labels = self._prepare_dataset(name)
        label_max = int(labels.max()) if labels is not None and np.any(labels > 0) else self.n_endmembers
        n_endmembers = int(self.n_endmembers or label_max)
        if n_endmembers < 1:
            raise ValueError(f"Unable to determine number of endmembers for {name}")

        rng = np.random.default_rng(self.random_state)
        if x.shape[0] > self.vca_sample_size:
            sample_indices = rng.choice(x.shape[0], size=self.vca_sample_size, replace=False)
            vca_input = x[sample_indices]
        else:
            vca_input = x

        model = FCLS(
            n_endmembers=n_endmembers,
            random_state=self.random_state,
            max_iter=self.fcls_max_iter,
            tol=self.fcls_tol,
            batch_size=self.fcls_batch_size,
        )
        self.logger.info(f"Fitting {name} with {n_endmembers} endmembers on {x.shape[0]} pixels")
        model.fit(vca_input)
        model.abundances_ = model.transform(x)

        abundances = model.abundances_.reshape(raw.shape[0], raw.shape[1], -1)
        reconstruction = model.reconstruct().reshape(raw.shape)
        reconstruction_error_map = np.mean((raw - reconstruction) ** 2, axis=2)
        dominant = np.argmax(abundances, axis=2)

        results = {
            "raw": raw,
            "x": x,
            "labels": labels,
            "model": model,
            "abundances": abundances,
            "reconstruction": reconstruction,
            "reconstruction_error_map": reconstruction_error_map,
            "dominant": dominant,
            "reconstruction_error": model.reconstruction_error(x),
            "abundance_sum_error": float(np.max(np.abs(abundances.sum(axis=2) - 1.0))),
            "mean_abundance": abundances.reshape(-1, abundances.shape[-1]).mean(axis=0),
        }

        if labels is not None and np.any(labels > 0):
            mask = labels > 0
            predicted = np.argmax(model.abundances_[mask], axis=1)
            target = labels[mask] - 1
            class_count = max(n_endmembers, int(target.max()) + 1)
            conf = confusion_matrix(target, predicted, labels=np.arange(class_count))
            results["labeled_mask"] = mask
            results["confusion_matrix"] = conf
            results["target_labels"] = target
            results["predicted_labels"] = predicted

        self.logger.info(
            f"{name}: reconstruction_error={results['reconstruction_error']:.6f}, abundance_sum_error={results['abundance_sum_error']:.6f}"
        )
        self._save_dataset_plots(name, results)
        return results

    def _save_dataset_plots(self, name, results):
        dataset_dir = self.output_dir / name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        model = results["model"]
        raw = results["raw"]
        abundances = results["abundances"]
        labels = results["labels"]
        reconstruction_error_map = results["reconstruction_error_map"]
        dominant = results["dominant"]

        band_axis = np.arange(raw.shape[-1])

        fig, ax = plt.subplots(figsize=(12, 5))
        for i, endmember in enumerate(model.endmembers_):
            ax.plot(band_axis, endmember, linewidth=1.5, label=f"EM {i + 1}")
        ax.set_xlabel("Band Index")
        ax.set_ylabel("Reflectance")
        ax.set_title(f"{name} Extracted Endmembers")
        ax.legend(ncol=2)
        fig.tight_layout()
        fig.savefig(dataset_dir / "01_endmember_spectra.pdf", bbox_inches="tight")
        plt.close(fig)

        n_endmembers = abundances.shape[-1]
        n_cols = min(3, n_endmembers)
        n_rows = int(np.ceil(n_endmembers / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.0 * n_rows))
        axes = np.atleast_1d(axes).ravel()
        for i in range(len(axes)):
            ax = axes[i]
            if i < n_endmembers:
                im = ax.imshow(abundances[:, :, i], cmap="magma")
                ax.set_title(f"Abundance {i + 1}")
                ax.set_xticks([])
                ax.set_yticks([])
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.axis("off")
        fig.suptitle(f"{name} Abundance Maps")
        fig.tight_layout()
        fig.savefig(dataset_dir / "02_abundance_maps.pdf", bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(dominant, cmap="tab20")
        ax.set_title(f"{name} Dominant Endmember Map")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(dataset_dir / "03_dominant_map.pdf", bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(reconstruction_error_map, cmap="viridis")
        ax.set_title(f"{name} Reconstruction Error Map")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(dataset_dir / "04_reconstruction_error.pdf", bbox_inches="tight")
        plt.close(fig)

        if labels is not None and "confusion_matrix" in results:
            conf = results["confusion_matrix"]
            fig, ax = plt.subplots(figsize=(7, 6))
            im = ax.imshow(conf, cmap="Blues")
            ax.set_title(f"{name} Label vs Dominant Endmember")
            ax.set_xlabel("Predicted Endmember")
            ax.set_ylabel("True Label")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            for i in range(conf.shape[0]):
                for j in range(conf.shape[1]):
                    ax.text(j, i, str(conf[i, j]), ha="center", va="center", color="black", fontsize=8)
            fig.tight_layout()
            fig.savefig(dataset_dir / "05_confusion_matrix.pdf", bbox_inches="tight")
            plt.close(fig)

            labeled_mask = labels > 0
            labeled_abundances = abundances.reshape(-1, abundances.shape[-1])[labeled_mask]
            class_ids = labels[labeled_mask]
            unique_classes = np.unique(class_ids)
            class_means = np.zeros((len(unique_classes), abundances.shape[-1]), dtype=np.float32)
            for i, class_id in enumerate(unique_classes):
                class_means[i] = labeled_abundances[class_ids == class_id].mean(axis=0)

            fig, ax = plt.subplots(figsize=(10, 5))
            x_positions = np.arange(len(unique_classes))
            bottom = np.zeros(len(unique_classes))
            for j in range(abundances.shape[-1]):
                ax.bar(x_positions, class_means[:, j], bottom=bottom, label=f"EM {j + 1}")
                bottom = bottom + class_means[:, j]
            ax.set_xticks(x_positions)
            ax.set_xticklabels([str(int(v)) for v in unique_classes])
            ax.set_xlabel("True Label")
            ax.set_ylabel("Mean Abundance")
            ax.set_title(f"{name} Mean Abundance by Label")
            ax.legend(ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left")
            fig.tight_layout()
            fig.savefig(dataset_dir / "06_label_abundance_summary.pdf", bbox_inches="tight")
            plt.close(fig)

    def train(self):
        if self.data_module is None:
            raise RuntimeError("Call setup() before train()")
        for name in self.dataset_names:
            self.results[name] = self._fit_single_dataset(name)
        return self.results

    def validate(self):
        validation_summary = {
            name: {
                "reconstruction_error": result["reconstruction_error"],
                "abundance_sum_error": result["abundance_sum_error"],
            }
            for name, result in self.results.items()
        }
        self.logger.info(f"Validation summary: {validation_summary}")
        return validation_summary

    def test(self):
        test_summary = {
            name: {
                "reconstruction_error": result["reconstruction_error"],
                "abundance_sum_error": result["abundance_sum_error"],
            }
            for name, result in self.results.items()
        }
        self.logger.info(f"Test summary: {test_summary}")
        return test_summary

    def run(self):
        self.setup()
        self.train()
        self.validate()
        self.test()
        return self.results


__all__ = ["HyMarsVCAFCLSExperiment"]
