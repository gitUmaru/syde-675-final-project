import numpy as np


def _project_simplex(values):
    values = np.asarray(values)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    n_rows, n_cols = values.shape
    sorted_values = np.sort(values, axis=1)[:, ::-1]
    cumulative = np.cumsum(sorted_values, axis=1)
    grid = np.arange(1, n_cols + 1, dtype=values.dtype)
    support = sorted_values - (cumulative - 1.0) / grid > 0
    rho = np.maximum(np.sum(support, axis=1) - 1, 0)
    theta = (cumulative[np.arange(n_rows), rho] - 1.0) / (rho + 1.0)
    projected = np.maximum(values - theta[:, None], 0.0)
    row_sums = projected.sum(axis=1, keepdims=True)
    projected = np.divide(
        projected,
        row_sums,
        out=np.full_like(projected, 1.0 / n_cols),
        where=row_sums > 0,
    )
    return projected


class VCA:
    def __init__(self, n_endmembers, random_state=None, snr_input=0.0, verbose=False):
        self.n_endmembers = int(n_endmembers)
        self.random_state = random_state
        self.snr_input = float(snr_input)
        self.verbose = verbose
        self.endmembers_ = None
        self.indices_ = None
        self.projection_ = None
        self.snr_ = None

    @staticmethod
    def estimate_snr(y, r_m, x):
        bands, pixels = y.shape
        p, _ = x.shape
        p_y = np.sum(y ** 2) / float(pixels)
        p_x = np.sum(x ** 2) / float(pixels) + np.sum(r_m ** 2)
        numerator = p_x - p / bands * p_y
        denominator = p_y - p_x
        if numerator <= 0 or denominator <= 0:
            return np.inf
        return 10.0 * np.log10(numerator / denominator)

    def fit(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError("Input data must be a 2D matrix")
        if x.shape[0] > x.shape[1]:
            x = x.T

        bands, pixels = x.shape
        if self.n_endmembers < 1 or self.n_endmembers > bands:
            raise ValueError("n_endmembers must be between 1 and the number of bands")

        rng = np.random.default_rng(self.random_state)
        y_mean = x.mean(axis=1, keepdims=True)
        y0 = x - y_mean
        covariance = (y0 @ y0.T) / float(pixels)
        basis, _, _ = np.linalg.svd(covariance, full_matrices=False)

        if self.snr_input == 0:
            x_p = basis[:, : self.n_endmembers].T @ y0
            snr = self.estimate_snr(x, y_mean, x_p)
        else:
            snr = self.snr_input

        self.snr_ = snr
        snr_threshold = 15.0 + 10.0 * np.log10(self.n_endmembers)

        if snr < snr_threshold:
            dim = self.n_endmembers - 1
            projection = basis[:, :dim]
            x_p = projection.T @ y0
            y_projected = projection @ x_p + y_mean
            c = np.sqrt(np.max(np.sum(x_p ** 2, axis=0)))
            y = np.vstack((x_p, c * np.ones((1, pixels), dtype=x_p.dtype)))
        else:
            dim = self.n_endmembers
            projection = basis[:, :dim]
            x_p = projection.T @ y0
            y_projected = projection @ x_p + y_mean
            mean_projection = x_p.mean(axis=1, keepdims=True)
            denominator = np.sum(x_p * mean_projection, axis=0, keepdims=True)
            denominator = np.where(np.abs(denominator) < np.finfo(x_p.dtype).eps, np.finfo(x_p.dtype).eps, denominator)
            y = x_p / denominator

        self.projection_ = y_projected.T
        indices = np.zeros(self.n_endmembers, dtype=int)
        a = np.zeros((self.n_endmembers, self.n_endmembers), dtype=x.dtype)
        a[-1, 0] = 1.0

        for i in range(self.n_endmembers):
            w = rng.random((self.n_endmembers, 1), dtype=x.dtype)
            f = w - a @ (np.linalg.pinv(a) @ w)
            norm = np.linalg.norm(f)
            if norm > 0:
                f = f / norm
            v = f.T @ y
            indices[i] = int(np.argmax(np.abs(v)))
            a[:, i] = y[:, indices[i]]

        self.indices_ = indices
        self.endmembers_ = self.projection_[indices]
        return self

    def transform(self, x=None):
        if self.endmembers_ is None:
            raise ValueError("The model must be fitted before calling transform")
        return self.endmembers_

    def fit_transform(self, x):
        self.fit(x)
        return self.transform()


class FCLS(VCA):
    def __init__(
        self,
        n_endmembers,
        random_state=None,
        snr_input=0.0,
        verbose=False,
        max_iter=50,
        tol=1e-5,
        batch_size=8192,
    ):
        super().__init__(n_endmembers=n_endmembers, random_state=random_state, snr_input=snr_input, verbose=verbose)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.batch_size = int(batch_size)
        self.abundances_ = None

    def _solve_abundances(self, x, endmembers):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError("Input data must be a 2D matrix")
        if x.shape[1] != endmembers.shape[1] and x.shape[0] == endmembers.shape[1]:
            x = x.T

        endmembers = np.asarray(endmembers, dtype=np.float32)
        if endmembers.ndim != 2:
            raise ValueError("Endmembers must be a 2D matrix")

        pixels, bands = x.shape
        n_endmembers = endmembers.shape[0]
        if endmembers.shape[1] != bands:
            raise ValueError("Endmembers and data must have the same number of bands")

        gram = endmembers @ endmembers.T
        gram = gram + np.eye(n_endmembers, dtype=gram.dtype) * np.finfo(gram.dtype).eps
        gram_inv = np.linalg.pinv(gram)
        lipschitz = 2.0 * float(np.linalg.eigvalsh(gram).max())
        step = 1.0 / (lipschitz + np.finfo(gram.dtype).eps)

        abundances = np.empty((pixels, n_endmembers), dtype=x.dtype)
        for start in range(0, pixels, self.batch_size):
            stop = min(start + self.batch_size, pixels)
            batch = x[start:stop]
            rhs = batch @ endmembers.T
            current = rhs @ gram_inv
            current = _project_simplex(current)

            for _ in range(self.max_iter):
                previous = current
                current = current - step * (current @ gram - rhs)
                current = _project_simplex(current)
                if np.max(np.abs(current - previous)) <= self.tol:
                    break

            abundances[start:stop] = current

        return abundances

    def fit(self, x):
        super().fit(x)
        self.abundances_ = self._solve_abundances(x, self.endmembers_)
        return self

    def transform(self, x=None):
        if self.endmembers_ is None:
            raise ValueError("The model must be fitted before calling transform")
        if x is None:
            if self.abundances_ is None:
                raise ValueError("No abundances are available")
            return self.abundances_
        return self._solve_abundances(x, self.endmembers_)

    def fit_transform(self, x):
        self.fit(x)
        return self.abundances_

    def reconstruct(self):
        if self.abundances_ is None or self.endmembers_ is None:
            raise ValueError("The model must be fitted before reconstruction")
        return self.abundances_ @ self.endmembers_

    def reconstruction_error(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError("Input data must be a 2D matrix")
        reconstruction = self.reconstruct()
        if x.shape != reconstruction.shape and x.shape[::-1] == reconstruction.shape:
            x = x.T
        return float(np.linalg.norm(x - reconstruction, ord="fro") ** 2 / (np.linalg.norm(x, ord="fro") ** 2 + np.finfo(x.dtype).eps))


class ClassicalBaseline(FCLS):
    pass


__all__ = ["VCA", "FCLS", "ClassicalBaseline"]
