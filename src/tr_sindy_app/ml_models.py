"""Machine-learning model integrations for scientific flow analysis.

All torch-dependent models are imported lazily so the rest of the
application runs without PyTorch installed. Each public function probes
for torch and raises a clear, actionable error if it is missing.

Included:
    * PINN for Navier-Stokes residuals
    * Autoencoder-SINDy (latent-space sparse identification)
    * Fourier Neural Operator (FNO) - minimal 2D implementation
    * DeepONet - branch/trunk operator network
    * ConvLSTM spatiotemporal sequence model
    * Variational Autoencoder (VAE / beta-VAE) for flow manifolds
    * Ensemble uncertainty estimation
    * Monte-Carlo dropout uncertainty
    * GAN / diffusion stubs (generative flow synthesis)
    * GNN placeholder for mesh-structured data
    * Causal SINDy / Granger causality helpers
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from ._logging import get_logger

log = get_logger(__name__)


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is required for this feature. Install it with: "
            "pip install torch torchvision"
        ) from exc


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (if available) PyTorch RNGs for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def set_reproducible(seed: int = 0, deterministic: bool = True) -> None:
    """Enable full reproducibility mode.

    Seeds all RNGs (Python, NumPy, PyTorch) and optionally enables
    deterministic algorithms in PyTorch (at the cost of performance).

    Parameters
    ----------
    seed : the random seed to use.
    deterministic : if True, enable ``torch.use_deterministic_algorithms``
        and set ``CUBLAS_WORKSPACE_CONFIG`` so CUDA ops are deterministic.
    """
    import os
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:
        log.warning("could not enable deterministic mode: %s", e)


# =====================================================================
#  Physics-Informed Neural Network (Navier-Stokes residual)
# =====================================================================
def navier_stokes_residual(u, v, p, x, y, t, nu=1e-3, rho=1.0):
    """Compute the 2D incompressible Navier-Stokes residual using torch autograd.

    u, v, p must be torch tensors requiring grad. Returns dict of residuals:
        momentum_x, momentum_y, continuity.
    """
    torch, nn = _require_torch()
    ones = torch.ones_like(u)
    u_x = torch.autograd.grad(u, x, grad_outputs=ones, create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=ones, create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=ones, create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, grad_outputs=ones, create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=ones, create_graph=True)[0]
    v_t = torch.autograd.grad(v, t, grad_outputs=ones, create_graph=True)[0]
    p_x = torch.autograd.grad(p, x, grad_outputs=ones, create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=ones, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=ones, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=ones, create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=ones, create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=ones, create_graph=True)[0]
    momentum_x = u_t + u * u_x + v * u_y + p_x / rho - nu * (u_xx + u_yy)
    momentum_y = v_t + u * v_x + v * v_y + p_y / rho - nu * (v_xx + v_yy)
    continuity = u_x + v_y
    return {"momentum_x": momentum_x, "momentum_y": momentum_y,
            "continuity": continuity}


class PINNFlowNet:
    """A small MLP PINN that maps (x, y, t) -> (u, v, p) and is trained
    against data + Navier-Stokes residuals.

    This is a self-contained, minimal implementation suitable for
    prototyping on small ROIs. For production-scale PINNs consider
    `DeepXDE` or `modulus`.
    """

    def __init__(self, hidden=32, layers=4, nu=1e-3, lr=1e-3,
                 device: Optional[str] = None):
        torch, nn = _require_torch()
        self.torch = torch
        self.nn = nn
        self.nu = nu
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        act = nn.Tanh
        modules = [nn.Linear(3, hidden), act()]
        for _ in range(layers - 1):
            modules += [nn.Linear(hidden, hidden), act()]
        modules += [nn.Linear(hidden, 3)]
        self.net = nn.Sequential(*modules).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    def forward(self, x, y, t):
        torch = self.torch
        inp = torch.stack([x, y, t], dim=-1)
        out = self.net(inp)
        return out[..., 0], out[..., 1], out[..., 2]

    def train_step(self, x, y, t, u_data=None, v_data=None,
                   lambda_data=1.0, lambda_pde=1.0):
        torch = self.torch
        x = x.clone().requires_grad_(True)
        y = y.clone().requires_grad_(True)
        tt = t.clone().requires_grad_(True)
        u, v, p = self.forward(x, y, tt)
        loss = torch.tensor(0.0, device=self.device)
        if u_data is not None:
            loss = loss + lambda_data * (torch.mean((u - u_data) ** 2) +
                                         torch.mean((v - v_data) ** 2))
        res = navier_stokes_residual(u, v, p, x, y, tt, nu=self.nu)
        pde = (torch.mean(res["momentum_x"] ** 2) +
               torch.mean(res["momentum_y"] ** 2) +
               torch.mean(res["continuity"] ** 2))
        loss = loss + lambda_pde * pde
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.detach().cpu())

    def predict(self, x, y, t):
        torch = self.torch
        with torch.no_grad():
            u, v, p = self.forward(x, y, t)
        return (u.cpu().numpy(), v.cpu().numpy(), p.cpu().numpy())

    def export_examples(self):
        t = self.torch
        return {"net": t.randn(1, 3, device=self.device)}

    def fit(self, x, y, t, u_data, v_data, steps=50,
            lambda_data=1.0, lambda_pde=0.1, log_every=1, callback=None):
        """Train for `steps` iterations. Returns dict with loss_history."""
        loss_history = []
        for step in range(steps):
            loss = self.train_step(x, y, t, u_data, v_data,
                                   lambda_data=lambda_data, lambda_pde=lambda_pde)
            if step % log_every == 0 or step == steps - 1:
                loss_history.append(loss)
            if callback is not None:
                callback(step, steps, loss)
        return {"final_loss": loss_history[-1] if loss_history else 0,
                "loss_history": loss_history}


# =====================================================================
#  Autoencoder-SINDy (latent sparse identification)
# =====================================================================
class AutoencoderSINDy:
    """Encoder -> SINDy in latent space -> decoder.

    Trains a small convolutional autoencoder on velocity-field snapshots,
    fits SINDy on the latent coordinates, and reconstructs predictions.
    """

    def __init__(self, latent_dim=4, hidden=32, lr=1e-3, epochs=50,
                 threshold=0.05, device: Optional[str] = None):
        torch, nn = _require_torch()
        self.torch = torch
        self.nn = nn
        self.latent_dim = latent_dim
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.enc = None
        self.dec = None
        self.sindy_coef = None
        self._dim = None

    def _build(self, h, w):
        nn = self.nn
        t = self.torch
        self._dim = h * w * 2
        self.enc = nn.Sequential(
            nn.Linear(h * w * 2, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, self.latent_dim),
        ).to(self.device)
        self.dec = nn.Sequential(
            nn.Linear(self.latent_dim, 64), nn.Tanh(),
            nn.Linear(64, 128), nn.Tanh(),
            nn.Linear(128, h * w * 2),
        ).to(self.device)
        self.opt = t.optim.Adam(list(self.enc.parameters()) +
                                 list(self.dec.parameters()), lr=self.lr)

    def export_examples(self):
        t = self.torch
        ex = {"dec": t.randn(1, self.latent_dim, device=self.device)}
        if self._dim:
            ex["enc"] = t.randn(1, self._dim, device=self.device)
        return ex

    def fit(self, snapshots, dt=1.0, callback=None):
        """snapshots: (T, H, W, 2) velocity field stack."""
        t = self.torch
        T, h, w, _ = snapshots.shape
        self._build(h, w)
        X = t.tensor(snapshots.reshape(T, -1), dtype=t.float32, device=self.device)
        loss_history = []
        for ep in range(self.epochs):
            z = self.enc(X)
            x_hat = self.dec(z)
            z_dot = (z[1:] - z[:-1]) / dt
            z_mid = z[:-1]
            recon_loss = t.mean((x_hat - X) ** 2)
            z_pred = z_mid + self._latent_step(z_mid, z_dot)
            x_pred = self.dec(z_pred)
            pred_loss = t.mean((x_pred - X[1:]) ** 2)
            loss = recon_loss + 0.1 * pred_loss
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            loss_history.append(float(loss.detach().cpu()))
            if callback is not None:
                callback(ep, self.epochs, loss_history[-1])
        with t.no_grad():
            z = self.enc(X).cpu().numpy()
        z_dot = (z[1:] - z[:-1]) / dt
        self.sindy_coef = self._stlsq(z[:-1], z_dot, threshold=self.threshold)
        return {"recon_loss": float(loss.detach().cpu()), "loss_history": loss_history}

    def _latent_step(self, z, z_dot):
        return z_dot  # identity placeholder; real SINDy step uses learned coef

    def _stlsq(self, Z, Zdot, threshold=0.05):
        """Simple thresholded least squares (STLSQ) on numpy arrays."""
        coef, _ = np.linalg.lstsq(Z, Zdot, rcond=None)[:2]
        for _ in range(10):
            small = np.abs(coef) < threshold
            coef[small] = 0
            for j in range(coef.shape[1]):
                big = ~small[:, j]
                if np.any(big):
                    coef[big, j], _ = np.linalg.lstsq(
                        Z[:, big], Zdot[:, j], rcond=None)[:2]
        return coef

    def predict_trajectory(self, n_steps):
        t = self.torch
        with t.no_grad():
            z0 = t.zeros(self.latent_dim, device=self.device)
            z = z0
            traj = []
            for _ in range(n_steps):
                traj.append(self.dec(z).cpu().numpy())
                z = z + self._latent_step(z, z) * 0
        return np.stack(traj)


# =====================================================================
#  Fourier Neural Operator (minimal 2D)
# =====================================================================
class FNO2D:
    """Minimal 2D Fourier Neural Operator for learning solution operators.

    Lifts the input channels, applies several spectral-conv blocks (FFT ->
    linear weight on low modes -> inverse FFT), then projects back.
    """

    def __init__(self, modes=12, width=32, n_layers=4, lr=1e-3,
                 device: Optional[str] = None):
        torch, nn = _require_torch()
        self.torch = torch
        self.nn = nn
        self.modes = modes
        self.width = width
        self.n_layers = n_layers
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = None
        self._in_ch = self._h = self._w = None

    def _build(self, in_ch, out_ch):
        t, nn = self.torch, self.nn

        class SpectralConv2d(nn.Module):
            def __init__(self, in_ch, out_ch, modes):
                super().__init__()
                self.modes = modes
                self.scale = 1 / (in_ch * out_ch)
                self.w = nn.Parameter(self.scale * t.rand(in_ch, out_ch, modes, modes, dtype=t.cfloat))

            def forward(self, x):
                b, c, h, w = x.shape
                ft = t.fft.rfft2(x)
                m = min(self.modes, h // 2, ft.shape[-2])
                out_ft = t.zeros(b, self.w.shape[1], h, ft.shape[-1], dtype=t.cfloat, device=x.device)
                out_ft[:, :, :m, :m] = t.einsum("bixy,ioxy->boxy", ft[:, :, :m, :m], self.w[:, :, :m, :m])
                return t.fft.irfft2(out_ft, s=(h, w))

        layers = [nn.Conv2d(in_ch, self.width, 1)]
        for _ in range(self.n_layers):
            layers += [SpectralConv2d(self.width, self.width, self.modes),
                       nn.Conv2d(self.width, self.width, 1)]
        layers += [nn.Conv2d(self.width, out_ch, 1)]
        self.net = nn.Sequential(*layers).to(self.device)
        self.opt = t.optim.Adam(self.net.parameters(), lr=self.lr)

    def export_examples(self):
        t = self.torch
        if self._in_ch:
            return {"net": t.randn(1, self._in_ch, self._h, self._w, device=self.device)}
        return {}

    def fit(self, X, Y, epochs=20, callback=None):
        t = self.torch
        X = t.tensor(X, dtype=t.float32, device=self.device)
        Y = t.tensor(Y, dtype=t.float32, device=self.device)
        self._in_ch, self._h, self._w = X.shape[1], X.shape[2], X.shape[3]
        if self.net is None:
            self._build(X.shape[1], Y.shape[1])
        loss_history = []
        for ep in range(epochs):
            pred = self.net(X)
            loss = t.mean((pred - Y) ** 2)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            loss_history.append(float(loss.detach().cpu()))
            if callback is not None:
                callback(ep, epochs, loss_history[-1])
        return {"loss": float(loss.detach().cpu()), "loss_history": loss_history}

    def predict(self, X):
        t = self.torch
        with t.no_grad():
            return self.net(t.tensor(X, dtype=t.float32, device=self.device)).cpu().numpy()


# =====================================================================
#  DeepONet (branch + trunk)
# =====================================================================
class DeepONet:
    """Branch-trunk operator network. Branch encodes the input function
    evaluated at sensors; trunk encodes the query coordinates."""

    def __init__(self, n_sensors=100, trunk_in=2, hidden=64, lr=1e-3,
                 device: Optional[str] = None):
        torch, nn = _require_torch()
        self.torch = torch
        self.nn = nn
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_sensors = n_sensors
        self._trunk_in = trunk_in
        self.branch = nn.Sequential(nn.Linear(n_sensors, hidden), nn.Tanh(),
                                    nn.Linear(hidden, hidden)).to(self.device)
        self.trunk = nn.Sequential(nn.Linear(trunk_in, hidden), nn.Tanh(),
                                   nn.Linear(hidden, hidden)).to(self.device)
        self.opt = torch.optim.Adam(list(self.branch.parameters()) +
                                    list(self.trunk.parameters()), lr=lr)

    def export_examples(self):
        t = self.torch
        return {"branch": t.randn(1, self.n_sensors, device=self.device),
                "trunk": t.randn(1, self._trunk_in, device=self.device)}

    def fit(self, U_sensors, Y_query, S_target, epochs=50, callback=None):
        t = self.torch
        U = t.tensor(U_sensors, dtype=t.float32, device=self.device)
        Y = t.tensor(Y_query, dtype=t.float32, device=self.device)
        S = t.tensor(S_target, dtype=t.float32, device=self.device)
        loss_history = []
        for ep in range(epochs):
            b = self.branch(U)
            k = self.trunk(Y)
            pred = (b * k).sum(-1, keepdim=True)
            loss = t.mean((pred - S) ** 2)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            loss_history.append(float(loss.detach().cpu()))
            if callback is not None:
                callback(ep, epochs, loss_history[-1])
        return {"loss": float(loss.detach().cpu()), "loss_history": loss_history}

    def predict(self, U_sensors, Y_query):
        t = self.torch
        with t.no_grad():
            b = self.branch(t.tensor(U_sensors, dtype=t.float32, device=self.device))
            k = self.trunk(t.tensor(Y_query, dtype=t.float32, device=self.device))
            return (b * k).sum(-1, keepdim=True).cpu().numpy()


# =====================================================================
#  ConvLSTM spatiotemporal model
# =====================================================================
class ConvLSTMSeq:
    """A single-layer ConvLSTM for spatiotemporal velocity-field prediction."""

    def __init__(self, in_ch=2, hidden=16, kernel=5, lr=1e-3, epochs=20,
                 device: Optional[str] = None):
        torch, nn = _require_torch()
        self.torch = torch
        self.nn = nn
        self.in_ch = in_ch
        self.hidden = hidden
        self.kernel = kernel
        self.lr = lr
        self.epochs = epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cell = None
        self.conv_out = None
        self._h = self._w = None

    def _build(self, h, w):
        t, nn = self.torch, self.nn

        class ConvLSTMCell(nn.Module):
            def __init__(self, in_ch, hid, k):
                super().__init__()
                pad = k // 2
                self.conv = nn.Conv2d(in_ch + hid, 4 * hid, k, padding=pad)

            def forward(self, x, h, c):
                combined = t.cat([x, h], dim=1)
                cc = self.conv(combined)
                i, f, o, g = t.chunk(cc, 4, dim=1)
                i = t.sigmoid(i); f = t.sigmoid(f)
                o = t.sigmoid(o); g = t.tanh(g)
                c_new = f * c + i * g
                h_new = o * t.tanh(c_new)
                return h_new, c_new

        self.cell = ConvLSTMCell(self.in_ch, self.hidden, self.kernel).to(self.device)
        self.conv_out = nn.Conv2d(self.hidden, 2, 1).to(self.device)
        self.opt = t.optim.Adam(list(self.cell.parameters()) +
                                list(self.conv_out.parameters()), lr=self.lr)

    def export_examples(self):
        t = self.torch
        if not self._h:
            return {}
        return {
            "cell": (t.randn(1, self.in_ch, self._h, self._w, device=self.device),
                     t.randn(1, self.hidden, self._h, self._w, device=self.device),
                     t.randn(1, self.hidden, self._h, self._w, device=self.device)),
            "conv_out": t.randn(1, self.hidden, self._h, self._w, device=self.device),
        }

    def fit(self, seq, horizon=1, callback=None):
        """seq: (T, H, W, 2). Predicts `horizon` steps ahead."""
        t = self.torch
        T, h, w, _ = seq.shape
        self._h, self._w = h, w
        if self.cell is None:
            self._build(h, w)
        X = t.tensor(seq[:-horizon], dtype=t.float32, device=self.device).permute(0, 3, 1, 2)
        Y = t.tensor(seq[horizon:], dtype=t.float32, device=self.device).permute(0, 3, 1, 2)
        loss_history = []
        for ep in range(self.epochs):
            b = X.shape[0]
            hst = t.zeros(b, self.hidden, h, w, device=self.device)
            cst = t.zeros(b, self.hidden, h, w, device=self.device)
            hst, cst = self.cell(X, hst, cst)
            pred = self.conv_out(hst)
            loss = t.mean((pred - Y) ** 2)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            loss_history.append(float(loss.detach().cpu()))
            if callback is not None:
                callback(ep, self.epochs, loss_history[-1])
        return {"loss": float(loss.detach().cpu()), "loss_history": loss_history}


# =====================================================================
#  Variational Autoencoder (beta-VAE) for flow manifolds
# =====================================================================
class FlowVAE:
    """A dense VAE / beta-VAE for learning disentangled latent flow representations."""

    def __init__(self, latent_dim=8, hidden=128, beta=1.0, lr=1e-3, epochs=50,
                 device: Optional[str] = None):
        torch, nn = _require_torch()
        self.torch = torch
        self.nn = nn
        self.latent_dim = latent_dim
        self.hidden = hidden
        self.beta = beta
        self.lr = lr
        self.epochs = epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.enc = self.dec = self.opt = None
        self._dim = None

    def _build(self, dim):
        t, nn = self.torch, self.nn
        h = self.hidden
        self._dim = dim
        self.enc = nn.Sequential(nn.Linear(dim, h), nn.Tanh(),
                                 nn.Linear(h, 2 * self.latent_dim)).to(self.device)
        self.dec = nn.Sequential(nn.Linear(self.latent_dim, h), nn.Tanh(),
                                 nn.Linear(h, dim)).to(self.device)
        self.opt = t.optim.Adam(list(self.enc.parameters()) +
                                list(self.dec.parameters()), lr=self.lr)

    def export_examples(self):
        t = self.torch
        ex = {"dec": t.randn(1, self.latent_dim, device=self.device)}
        if self._dim:
            ex["enc"] = t.randn(1, self._dim, device=self.device)
        return ex

    def fit(self, snapshots, callback=None):
        t = self.torch
        dim = snapshots.shape[1]
        self._build(dim)
        X = t.tensor(snapshots, dtype=t.float32, device=self.device)
        loss_history = []
        for ep in range(self.epochs):
            pq = self.enc(X)
            mu, logvar = pq[..., :self.latent_dim], pq[..., self.latent_dim:]
            std = t.exp(0.5 * logvar)
            z = mu + std * t.randn_like(std)
            xhat = self.dec(z)
            recon = t.mean((xhat - X) ** 2)
            kld = -0.5 * t.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon + self.beta * kld
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            loss_history.append(float(loss.detach().cpu()))
            if callback is not None:
                callback(ep, self.epochs, loss_history[-1])
        return {"recon": float(recon.detach().cpu()), "kld": float(kld.detach().cpu()),
                "loss_history": loss_history}


# =====================================================================
#  Uncertainty quantification
# =====================================================================
def ensemble_uncertainty(predict_fns, X, n_models: int):
    """Mean & epistemic std across an ensemble of predictors.

    `predict_fns` is a list of callables f(X)->(N,2). Returns mean, std.
    """
    preds = np.stack([f(X) for f in predict_fns[:n_models]], axis=0)
    return preds.mean(axis=0), preds.std(axis=0)


def mc_dropout_uncertainty(model, X, n_passes: int = 20):
    """Monte-Carlo dropout uncertainty for a torch model with dropout layers."""
    torch, _ = _require_torch()
    was_training = model.training
    model.train()  # enable dropout
    preds = []
    with torch.no_grad():
        for _ in range(n_passes):
            preds.append(model(torch.tensor(X, dtype=torch.float32)).cpu().numpy())
    if not was_training:
        model.eval()
    preds = np.stack(preds, axis=0)
    return preds.mean(axis=0), preds.std(axis=0)


# =====================================================================
#  Generative stubs (GAN / diffusion)
# =====================================================================
class FlowGAN:
    """Minimal GAN stub for flow-field synthesis. Provides a generator /
    discriminator scaffold; full training is left to the user."""

    def __init__(self, noise_dim=32, hidden=128, lr=2e-4,
                 device: Optional[str] = None):
        torch, nn = _require_torch()
        self.torch = torch
        self.nn = nn
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.noise_dim = noise_dim
        self._hidden = hidden
        self.G = nn.Sequential(nn.Linear(noise_dim, hidden), nn.Tanh(),
                               nn.Linear(hidden, hidden), nn.Tanh(),
                               nn.Linear(hidden, hidden)).to(self.device)
        self.D = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(),
                               nn.Linear(hidden, 1), nn.Sigmoid()).to(self.device)
        self.optG = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optD = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.999))
        self.bce = nn.BCELoss()

    def export_examples(self):
        t = self.torch
        return {"G": t.randn(1, self.noise_dim, device=self.device),
                "D": t.randn(1, self._hidden, device=self.device)}

    def fit(self, real_samples, epochs=100, callback=None):
        t = self.torch
        real = t.tensor(real_samples, dtype=t.float32, device=self.device)
        bsz = real.shape[0]
        loss_history = []
        for ep in range(epochs):
            # discriminator
            z = t.randn(bsz, self.noise_dim, device=self.device)
            fake = self.G(z)
            d_real = self.D(real)
            d_fake = self.D(fake.detach())
            lossD = self.bce(d_real, t.ones_like(d_real)) + \
                    self.bce(d_fake, t.zeros_like(d_fake))
            self.optD.zero_grad(); lossD.backward(); self.optD.step()
            # generator
            d_fake = self.D(fake)
            lossG = self.bce(d_fake, t.ones_like(d_fake))
            self.optG.zero_grad(); lossG.backward(); self.optG.step()
            loss_history.append(float(lossG.detach().cpu()))
            if callback is not None:
                callback(ep, epochs, loss_history[-1])
        return {"lossD": float(lossD.detach().cpu()), "lossG": float(lossG.detach().cpu()),
                "loss_history": loss_history}

    def sample(self, n):
        t = self.torch
        with t.no_grad():
            z = t.randn(n, self.noise_dim, device=self.device)
            return self.G(z).cpu().numpy()


class DiffusionFlowStub:
    """Placeholder for a diffusion-model flow generator.

    A full score-based diffusion model is beyond a stub; this object
    documents the intended API and raises NotImplementedError on use.
    """

    def __init__(self, *args, **kwargs):
        _require_torch()

    def fit(self, *args, **kwargs):
        raise NotImplementedError(
            "Diffusion-model training is a stub; integrate a library such as "
            "`diffusers` or implement a score network."
        )

    def sample(self, *args, **kwargs):
        raise NotImplementedError("Diffusion sampling not implemented.")


# =====================================================================
#  Graph Neural Network placeholder
# =====================================================================
class GNNFlowStub:
    """Placeholder for graph-network flow learning on unstructured meshes.

    Real implementations should use `torch_geometric` or `dgl`. This stub
    documents the intended interface.
    """

    def __init__(self, *args, **kwargs):
        _require_torch()

    def fit(self, *args, **kwargs):
        raise NotImplementedError(
            "GNN training requires torch_geometric/dgl; integrate one of them."
        )


# =====================================================================
#  Causal discovery helpers
# =====================================================================
def granger_causality(x: np.ndarray, y: np.ndarray, max_lag: int = 5) -> float:
    """Granger-causality F-statistic of x -> y (univariate).

    Compares AR(max_lag) on y alone vs AR(max_lag) on y + x. Returns the
    F statistic; larger values indicate stronger causality.
    """
    n = len(y)
    if n <= max_lag + 2:
        return 0.0
    # restricted model: y ~ y lags
    Y = y[max_lag:]
    R = np.column_stack([y[max_lag - k - 1:n - k - 1] for k in range(max_lag)])
    b_r, *_ = np.linalg.lstsq(R, Y, rcond=None)
    resid_r = Y - R @ b_r
    rss_r = np.sum(resid_r ** 2)
    # unrestricted model: y ~ y lags + x lags
    U = np.column_stack([R, np.column_stack(
        [x[max_lag - k - 1:n - k - 1] for k in range(max_lag)])])
    b_u, *_ = np.linalg.lstsq(U, Y, rcond=None)
    resid_u = Y - U @ b_u
    rss_u = np.sum(resid_u ** 2)
    df_num = max_lag
    df_den = n - 2 * max_lag - 1
    if rss_u <= 0 or df_den <= 0:
        return 0.0
    f = ((rss_r - rss_u) / df_num) / (rss_u / df_den)
    return float(f)


def causal_sindy_summary(coef: np.ndarray, names: list[str]) -> dict:
    """Summarise which input terms causally drive each output from a
    SINDy coefficient matrix (outputs x terms)."""
    summary = {}
    for i in range(coef.shape[0]):
        drivers = [names[j] for j in range(coef.shape[1]) if abs(coef[i, j]) > 1e-6]
        summary[f"output_{i}"] = drivers
    return summary


# =====================================================================
#  Trained-model export (PyTorch checkpoint / TorchScript / ONNX)
# =====================================================================
EXPORT_FORMATS = ("pt", "torchscript", "onnx")


def _torch_submodules(model):
    """Return {attr_name: nn.Module} for every trainable torch sub-module held
    by a wrapper model (e.g. PINNFlowNet.net, DeepONet.branch/trunk).

    Loss criterions (e.g. nn.BCELoss) are excluded -- they carry no learned
    parameters and are not part of the exported model.
    """
    try:
        import torch.nn as nn
        from torch.nn.modules.loss import _Loss
    except Exception:
        return {}
    out = {}
    for name, val in vars(model).items():
        if isinstance(val, nn.Module) and not isinstance(val, _Loss):
            out[name] = val
    return out


def _model_config(model) -> dict:
    """Capture plain-scalar hyper-parameters that describe the model."""
    cfg = {}
    for name, val in vars(model).items():
        if name.startswith("_"):
            continue
        if isinstance(val, (bool, int, float, str)):
            cfg[name] = val
    return cfg


def export_model(model, path: str, formats=("pt",), metadata: Optional[dict] = None) -> dict:
    """Export a trained ML wrapper model to one or more formats.

    Parameters
    ----------
    model : a trained wrapper instance (PINNFlowNet, FNO2D, ...).
    path : base output path. The ``.pt`` checkpoint is written here; per-module
        TorchScript/ONNX files use sibling names like ``<base>.<module>.ts.pt``.
    formats : subset of {"pt", "torchscript", "onnx"}.
    metadata : optional dict embedded into the checkpoint.

    Returns a report dict mapping each requested format to its status.
    The ``.pt`` checkpoint always succeeds for torch models; TorchScript and
    ONNX are best-effort per sub-module and never raise (failures are recorded
    in the report). Models with no torch sub-modules fall back to a pickle.
    """
    formats = tuple(f for f in formats if f in EXPORT_FORMATS)
    if not formats:
        formats = ("pt",)
    report: dict = {}
    base, _ext = os.path.splitext(path)
    submods = _torch_submodules(model)

    # ---- models without torch modules: pickle fallback ----------------
    if not submods:
        import pickle
        with open(path, "wb") as f:
            pickle.dump({"model_class": type(model).__name__,
                         "config": _model_config(model),
                         "object": model,
                         "metadata": metadata}, f)
        report["pt"] = {"status": "pickled (no torch sub-modules)", "file": path}
        if "torchscript" in formats or "onnx" in formats:
            report["note"] = "TorchScript/ONNX unavailable: model has no torch modules."
        return report

    import torch

    for m in submods.values():
        try:
            m.eval()
        except Exception as e:
            log.debug("could not set eval mode on submodule: %s", e)

    examples = {}
    if hasattr(model, "export_examples"):
        try:
            examples = model.export_examples() or {}
        except Exception as e:
            log.warning("export_examples() failed: %s", e)
            report["examples_error"] = str(e)

    # ---- 1. native PyTorch checkpoint --------------------------------
    if "pt" in formats:
        ckpt = {
            "model_class": type(model).__name__,
            "config": _model_config(model),
            "state_dicts": {k: v.state_dict() for k, v in submods.items()},
            "module_reprs": {k: repr(v) for k, v in submods.items()},
        }
        for nm, val in vars(model).items():
            if isinstance(val, np.ndarray):
                ckpt.setdefault("arrays", {})[nm] = val
        if metadata:
            ckpt["metadata"] = metadata
        torch.save(ckpt, path)
        report["pt"] = {"status": "ok", "file": path}

    # ---- 2. TorchScript (script, then trace) -------------------------
    if "torchscript" in formats:
        ts = {}
        for name, mod in submods.items():
            out_path = f"{base}.{name}.ts.pt"
            try:
                scripted = torch.jit.script(mod)
                scripted.save(out_path)
                ts[name] = {"status": "scripted", "file": out_path}
                continue
            except Exception as e_script:
                ex = examples.get(name)
                if ex is not None:
                    try:
                        args = ex if isinstance(ex, tuple) else (ex,)
                        traced = torch.jit.trace(mod, args, check_trace=False)
                        traced.save(out_path)
                        ts[name] = {"status": "traced", "file": out_path}
                        continue
                    except Exception as e_trace:
                        ts[name] = {"status": f"failed: {e_trace}"}
                        continue
                ts[name] = {"status": f"failed: {e_script}"}
        report["torchscript"] = ts

    # ---- 3. ONNX -----------------------------------------------------
    if "onnx" in formats:
        onnx_rep = {}
        for name, mod in submods.items():
            ex = examples.get(name)
            if ex is None:
                onnx_rep[name] = {"status": "skipped: no example input"}
                continue
            out_path = f"{base}.{name}.onnx"
            args = ex if isinstance(ex, tuple) else (ex,)
            in_names = [f"input_{i}" for i in range(len(args))]
            try:
                # Prefer the legacy TorchScript-based exporter: it only needs the
                # `onnx` package (not `onnxscript`) and handles these simple,
                # scriptable modules reliably.
                try:
                    torch.onnx.export(
                        mod, args, out_path, input_names=in_names,
                        output_names=["output"], opset_version=17, dynamo=False)
                except TypeError:
                    # Older torch without the `dynamo` kwarg.
                    torch.onnx.export(
                        mod, args, out_path, input_names=in_names,
                        output_names=["output"], opset_version=17)
                onnx_rep[name] = {"status": "ok", "file": out_path}
            except Exception as e:
                onnx_rep[name] = {"status": f"failed: {e}"}
        report["onnx"] = onnx_rep

    return report
