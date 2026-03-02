from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict

import numpy as np

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None


class DistributionWrapper:
    """
    Wrapper de distribuciones univariadas usado por el costo bayesiano.

    En este proyecto se cargan dos instancias:
    - `correct`: distribuciones calibradas con pares (track, deteccion) que SI eran match real.
    - `incorrect`: distribuciones calibradas con pares que NO eran match real.

    Luego, en `costs.py`, ambas probabilidades se comparan con un likelihood ratio para
    obtener `prob_match` posterior.
    """

    def __init__(self) -> None:
        self.distributions: Dict[str, dict] = {}
        self.feature_types: Dict[str, str] = {}

    def load(self, filepath: str) -> None:
        """
        Carga JSON de distribuciones con estructura:
        {
          "feature_types": {...},
          "distributions": {
            "<feature_name>": {"type": "...", ...}
          }
        }
        """
        payload = json.loads(Path(filepath).read_text())
        self.distributions = payload.get("distributions", {}) or {}
        self.feature_types = payload.get("feature_types", {}) or {}

    def get_probability(self, feature_name: str, value):
        """
        Evalua la densidad (PDF) para una feature.

        Nota:
        - Puede recibir escalar o vector.
        - El resultado se clippea a un minimo (`1e-12`) para evitar ceros que
          romperian el `-log(prob)` del costo bayesiano.
        """
        if feature_name not in self.distributions:
            raise ValueError(f"Feature '{feature_name}' has not been loaded")

        info = self.distributions[feature_name]
        dist_type = str(info.get("type", "")).lower().strip()
        is_scalar = np.isscalar(value)
        x = np.atleast_1d(value).astype(np.float64)

        if dist_type == "normal":
            out = self._normal_pdf(x, info.get("params", {}))
        elif dist_type == "beta":
            out = self._beta_pdf(x, info)
        elif dist_type == "t":
            out = self._t_pdf(x, info.get("params", {}))
        elif dist_type == "gmm":
            out = self._gmm_pdf(x, info)
        elif dist_type == "negexp":
            out = self._negexp_pdf(x, info)
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")

        out = np.asarray(out, dtype=np.float64)
        out = np.clip(out, 1e-12, np.inf)
        return float(out[0]) if is_scalar else out

    @staticmethod
    def _normal_pdf(x: np.ndarray, params: dict) -> np.ndarray:
        mean = float(params.get("mean", 0.0))
        std = max(float(params.get("std", 1.0)), 1e-8)
        if scipy_stats is not None:
            return scipy_stats.norm.pdf(x, loc=mean, scale=std)
        z = (x - mean) / std
        return (1.0 / (std * math.sqrt(2.0 * math.pi))) * np.exp(-0.5 * z * z)

    @staticmethod
    def _beta_pdf(x: np.ndarray, info: dict) -> np.ndarray:
        params = info.get("params", {}) or {}
        norm = info.get("normalization", {}) or {}
        dmin = float(norm.get("min", 0.0))
        dmax = float(norm.get("max", 1.0))
        if dmax <= dmin:
            return np.ones_like(x, dtype=np.float64)

        xn = (x - dmin) / (dmax - dmin)
        xn = np.clip(xn, 0.0, 1.0)

        a = max(float(params.get("a", 1.0)), 1e-8)
        b = max(float(params.get("b", 1.0)), 1e-8)
        loc = float(params.get("loc", 0.0))
        scale = max(float(params.get("scale", 1.0)), 1e-8)

        if scipy_stats is not None:
            out = scipy_stats.beta.pdf(xn, a, b, loc=loc, scale=scale)
        else:
            y = (xn - loc) / scale
            y = np.clip(y, 1e-8, 1.0 - 1e-8)
            # B(a,b) via log-gamma for numerical stability.
            log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
            out = np.exp((a - 1.0) * np.log(y) + (b - 1.0) * np.log(1.0 - y) - log_beta) / scale

        cutoff = params.get("cutoff", None)
        cutoff_val = params.get("cutoff_value", None)
        if cutoff is not None and cutoff_val is not None:
            out[xn > float(cutoff)] = float(cutoff_val)
        return out

    @staticmethod
    def _t_pdf(x: np.ndarray, params: dict) -> np.ndarray:
        df = max(float(params.get("df", 1.0)), 1e-8)
        loc = float(params.get("loc", 0.0))
        scale = max(float(params.get("scale", 1.0)), 1e-8)
        if scipy_stats is not None:
            return scipy_stats.t.pdf(x, df=df, loc=loc, scale=scale)

        z = (x - loc) / scale
        c = math.exp(
            math.lgamma((df + 1.0) * 0.5)
            - math.lgamma(df * 0.5)
            - 0.5 * math.log(df * math.pi)
            - math.log(scale)
        )
        return c * np.power(1.0 + (z * z) / df, -0.5 * (df + 1.0))

    @staticmethod
    def _gmm_pdf(x: np.ndarray, info: dict) -> np.ndarray:
        weights = np.asarray(info.get("weights", []), dtype=np.float64).reshape(-1)
        means = np.asarray(info.get("means", []), dtype=np.float64).reshape(-1)
        covs = np.asarray(info.get("covariances", []), dtype=np.float64).reshape(-1)

        if weights.size == 0 or means.size == 0 or covs.size == 0:
            return np.ones_like(x, dtype=np.float64)

        n = int(min(weights.size, means.size, covs.size))
        weights = weights[:n]
        means = means[:n]
        covs = np.clip(covs[:n], 1e-8, np.inf)
        weights = np.clip(weights, 1e-12, np.inf)
        weights = weights / np.sum(weights)

        # 1D Gaussian mixture density.
        x_col = x.reshape(-1, 1)
        std = np.sqrt(covs).reshape(1, -1)
        mean = means.reshape(1, -1)
        z = (x_col - mean) / std
        comp = (1.0 / (std * math.sqrt(2.0 * math.pi))) * np.exp(-0.5 * z * z)
        return np.sum(comp * weights.reshape(1, -1), axis=1)

    @staticmethod
    def _negexp_pdf(x: np.ndarray, info: dict) -> np.ndarray:
        params = info.get("params", {}) or {}
        norm = info.get("normalization", {}) or {}
        k = max(float(params.get("k", 1.0)), 1e-8)
        dmin = float(norm.get("min", 0.0))
        dmax = float(norm.get("max", 1.0))
        if dmax <= dmin:
            return np.ones_like(x, dtype=np.float64)

        xn = (x - dmin) / (dmax - dmin)
        xn = np.clip(xn, 0.0, 1.0)
        denom = max(1.0 - math.exp(-k), 1e-12)
        unit_pdf = (k * np.exp(-k * (1.0 - xn))) / denom
        return unit_pdf / (dmax - dmin)
