"""
zkit.mwigner.transform — research-grade *marginal* Wigner transform
(ħ = 1) and the associated phase-space metrics, ported from
wigner_quantum.py and integrated into zkit.  No plotting here; figures
live in zkit.mwigner.analysis.

The x-marginal Wigner function is

    W(x, p_x) = (dx*dy/π) ∫∫ ψ*(x+ξ, y) ψ(x-ξ, y) e^{+2 i p_x ξ} dy dξ

evaluated on a uniform momentum grid p_x ∈ [-p_max, p_max] with
spacing dp.  The wavefunction is zero-padded by M_max on each side
before the sliding-window transform, so the returned W covers the whole
x-domain and no tail is cropped.
"""

import warnings

import h5py
import numpy as np
import torch


def get_device():
    """Best available torch device (cuda > mps > cpu)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def marginal_wigner(
    psi_np,
    dxs,
    axis=0,
    device=None,
    chunk_size=64,
    M_max=None,
    window_type="none",
    warn_on_alias=True,
):
    """Marginal Wigner transform of an N-dimensional wavefunction.

    Computes the Wigner function marginalised over every axis EXCEPT
    `axis` (the "volume integration" over all transverse coordinates):

        W(x_a, p_a) = (prod_i dx_i / pi)
                       int ... int  psi*(x_a+xi, x_T) psi(x_a-xi, x_T)
                       x e^{+2 i p_a xi}  d xi prod_{T!=a} dx_T

    For a 2D wavefunction with axis=0 this reduces exactly to the
    x-marginal Wigner used throughout zkit.mwigner; for a 3D
    wavefunction it integrates out both transverse coordinates, giving
    the true 3D marginal Wigner (no separate 3D routine needed).

    Parameters
    ----------
    psi_np : np.ndarray
        Spatial wavefunction, shape (n_0, ..., n_{d-1}), any d >= 2.
    dxs : sequence of float
        Grid spacings, one per spatial axis (length == psi_np.ndim).
    axis : int
        Active axis to keep (marginalise over all others).
    ... (chunk_size, M_max, window_type, warn_on_alias as in the 2D
         routine; see get_wigner_research for semantics).

    Returns
    -------
    W : (n_axis, window_size) np.ndarray  (normalised by prod dx / pi)
    p_v : (window_size,) np.ndarray        momentum grid (increasing, dp>0)
    """
    if device is None:
        device = get_device()

    psi_np = np.asarray(psi_np)
    ndim = psi_np.ndim
    if ndim < 2:
        raise ValueError("marginal_wigner: need at least 2 spatial dims")
    if len(dxs) != ndim:
        raise ValueError("marginal_wigner: len(dxs) must equal psi.ndim")
    dxs = [float(d) for d in dxs]
    a = axis % ndim
    dxa = dxs[a]
    # product of all grid spacings (the volume element)
    vol = float(np.prod(dxs))

    if not np.all(np.isfinite(psi_np)):
        warnings.warn(
            "marginal_wigner: input contains non-finite values; "
            "returning a zero Wigner with the correct momentum grid.",
            RuntimeWarning,
        )
        na = psi_np.shape[a]
        window_size = min(2 * (na // 4) + 1, na)
        k = np.arange(window_size) - window_size // 2
        p_v = np.pi * k / (window_size * dxa)
        return np.zeros((na, window_size), dtype=np.float64), p_v

    na = psi_np.shape[a]
    if M_max is None:
        M_max = na // 4
    M_max_user = M_max
    M_max = int(min(max(M_max, 1), na - 1))

    # Resolution guard: estimate the x_a-coherence support from the
    # active-axis marginal density (summed over all transverse axes).
    rho_a = np.sum(np.abs(psi_np) ** 2, axis=tuple(i for i in range(ndim) if i != a))
    if warn_on_alias:
        s = rho_a.sum()
        if s > 0:
            xx = np.arange(na) * dxa
            cdf = np.cumsum(rho_a) / s
            lo = xx[np.searchsorted(cdf, 0.01)]
            hi = xx[np.searchsorted(cdf, 0.99)]
            support = hi - lo
            req = 0.7 * support
            if M_max * dxa < 0.9 * req:
                warnings.warn(
                    "marginal_wigner: M_max=%d (xi-window %.3g "
                    "a.u.) is smaller than required (~%.3g a.u.) for "
                    "this state's axis-%d coherence; off-diagonal "
                    "coherences are truncated and the Wigner/"
                    "negativity will be inaccurate. Increase M_max "
                    "(or pass M_max=None for the default na//4)."
                    % (M_max_user, M_max * dxa, req, a),
                    RuntimeWarning,
                )

    if device.type == "mps":
        np_dtype, f_dtype = np.complex64, torch.float32
    else:
        np_dtype, f_dtype = np.complex128, torch.float64

    # ── zero-pad ONLY along the active axis ─────────────────────────────
    psi_tensor = torch.from_numpy(psi_np.astype(np_dtype)).to(device)
    pad_shape = list(psi_tensor.shape)
    pad_shape[a] = na + 2 * M_max
    psi_padded = torch.zeros(pad_shape, dtype=psi_tensor.dtype, device=device)
    sl = [slice(None)] * ndim
    sl[a] = slice(M_max, M_max + na)
    psi_padded[tuple(sl)] = psi_tensor

    window_size = 2 * M_max + 1

    if window_type == "hann":
        win = torch.hann_window(window_size, periodic=False, dtype=f_dtype, device=device)
    else:
        win = torch.ones(window_size, dtype=f_dtype, device=device)

    weights = torch.ones(window_size, dtype=f_dtype, device=device)
    weights[0] = weights[-1] = 0.5
    # filter_vec must broadcast along ONLY the window axis, which `unfold`
    # appends as the LAST dimension (..., window_size).  Singleton everywhere
    # else so it spans the chunk and all transverse axes.
    filter_vec = (win * weights).reshape((1,) * ndim + (window_size,))
    # -> (1, ..., 1, window_size); multiplies each window column uniformly.

    # Transverse axes are everything except the active axis.  The kernel
    # is summed over all of them (the "volume integration").
    trans_axes = tuple(i for i in range(ndim) if i != a)

    # Memory-safe chunking over the active-axis output index.
    peak_budget = 120_000_000
    # size of one (x_chunk, window, *transverse) complex tensor:
    trans_elems = int(np.prod([psi_np.shape[i] for i in trans_axes])) if trans_axes else 1
    x_chunk = int(max(1, min(chunk_size, peak_budget // (3 * window_size * max(trans_elems, 1)))))

    W_out = torch.zeros((na, window_size), dtype=f_dtype, device=device)
    # Move the active axis to position 0 so unfold() can slide along it.
    psi_padded = psi_padded.permute([a] + [i for i in range(ndim) if i != a])
    # unfold along the active (now dim 0) axis with window `window_size`,
    # step 1 -> (na, *transverse, window_size) (unfold appends the window
    # as the LAST dimension).
    win_c_all = psi_padded.unfold(0, window_size, 1)

    # Memory-safe chunking over the active-axis output index.
    peak_budget = 120_000_000
    trans_elems = int(np.prod([psi_np.shape[i] for i in trans_axes])) if trans_axes else 1
    x_chunk = int(max(1, min(chunk_size, peak_budget // (3 * window_size * max(trans_elems, 1)))))
    for x0 in range(0, na, x_chunk):
        x1 = min(x0 + x_chunk, na)
        win_c = win_c_all[x0:x1]  # (x_chunk, *transverse, window_size)

        # === KERNEL: Psi*(x+xi) * Psi(x-xi) ===
        # win_c has shape (x_chunk, *transverse, window_size); the window is
        # the LAST axis (from unfold), transverse are dims 1..ndim-1.
        psi_plus = win_c
        psi_minus = torch.flip(win_c, dims=[-1])
        kernel = (psi_plus.conj() * psi_minus * filter_vec).sum(
            dim=tuple(range(1, win_c.ndim - 1))
        )  # -> (x_chunk, window_size)

        kernel = torch.fft.ifftshift(kernel, dim=-1)
        W_rows = torch.fft.fft(kernel, dim=-1)
        W_rows = torch.fft.fftshift(W_rows, dim=-1).real

        W_out[x0:x1] = W_rows

    # Momentum grid (increasing axis, dp > 0).
    k = np.arange(window_size) - window_size // 2
    p_v = np.pi * k / (window_size * dxa)  # increasing axis, dp > 0
    W = torch.flip(W_out, dims=[1]).cpu().numpy().astype(np.float64) * (vol / np.pi)

    # Aliasing guard on the active-axis marginal momentum.
    p_Ny = np.pi / (2.0 * dxa)
    if warn_on_alias and W.size:
        dp = p_v[1] - p_v[0]
        wt = np.clip(W, 0.0, None) + 1e-300
        p_mean = np.sum(wt * p_v[None, :], axis=1) * dp
        p_var = np.sum(wt * (p_v[None, :] - p_mean[:, None]) ** 2, axis=1) * dp
        p_rms_max = np.sqrt(np.max(p_var))
        if p_rms_max > 0.9 * p_Ny:
            warnings.warn(
                "marginal_wigner: characteristic marginal momentum "
                "(p_rms ~ %.3g) approaches the Nyquist limit %.3g; "
                "refine dx or increase M_max to resolve it without "
                "aliasing." % (p_rms_max, p_Ny),
                RuntimeWarning,
            )

    return W, p_v


def get_wigner_research(
    psi_2d_np: np.ndarray,
    dx: float,
    dy: float,
    device: torch.device = None,
    chunk_size: int = 64,
    M_max: int = None,
    window_type: str = "none",
    warn_on_alias: bool = True,
) -> tuple:
    """Research-grade 2D marginal Wigner transform (ħ = 1).

    Thin wrapper over :func:`marginal_wigner` for the common 2D case
    (active axis 0).  Computes

        W(x, p_x) = (dx*dy/pi) int int psi*(x+xi, y) psi(x-xi, y)
                                       e^{+2 i p_x xi} dy dxi

    over the FULL input domain (the wavefunction is zero-padded by
    M_max on each side along x so no tail is cropped).  See
    :func:`marginal_wigner` for the full N-dimensional generalisation
    (including 3D volume integration).
    """
    return marginal_wigner(
        psi_2d_np,
        (dx, dy),
        axis=0,
        device=device,
        chunk_size=chunk_size,
        M_max=M_max,
        window_type=window_type,
        warn_on_alias=warn_on_alias,
    )


def load_psi(file_path: str, snap_idx: int) -> np.ndarray:
    """Read snapshot `snap_idx` from an h5 wavefunction file as complex.

    Handles both the 2D layout ``(nx, ny, 2)`` and the general N-D
    PETSc layout ``(..., 2)`` (the trailing axis is [real, imag]).
    """
    with h5py.File(file_path, "r") as f:
        data = f["wavefunction"][snap_idx]
        if data.ndim >= 2 and data.shape[-1] == 2:
            return data[..., 0] + 1j * data[..., 1]
        # real-only fallback
        return np.array(data, dtype=np.complex128)


def measure_wigner_negativity(W: np.ndarray, dx: float, dp: float, normalize: bool = True) -> float:
    """Wigner negativity (volume of the negative regions)."""
    if W.size == 0 or np.all(W == 0):
        return 0.0
    if normalize:
        norm = np.sum(W) * dx * dp
        if norm <= 0:
            return 0.0
        W = W / norm
    negative_parts = W[W < 0]
    if negative_parts.size == 0:
        return 0.0
    return np.sum(np.abs(negative_parts)) * dx * dp


def verify_purities_and_entropy(psi, W, dxs, dp, axis=0, M_max=None):
    """Global norm, reduced purities, and Von Neumann entanglement entropy.

    Works for an N-dimensional wavefunction.  The reduced (axis-marginal)
    density matrix is formed over the FULL active-axis domain
    (independent of M_max) by tracing out every transverse coordinate:

        rho[a, b] = (prod_{T} dT) * sum_T psi[a, T] psi*[b, T]

    Parameters
    ----------
    psi : np.ndarray
        Spatial wavefunction, shape (n_0, ..., n_{d-1}).
    W : np.ndarray
        Marginal Wigner from :func:`marginal_wigner` (shape (n_axis, n_p)).
    dxs : float or sequence of float
        Grid spacings; a single float or one per spatial axis.
    dp : float
        Momentum-grid spacing of W.
    axis : int
        Active axis (the one NOT traced out).
    """
    psi = np.asarray(psi)
    ndim = psi.ndim
    if np.isscalar(dxs) or (hasattr(dxs, "__len__") and len(dxs) == 1):
        dxs = [float(dxs)] * ndim
    else:
        dxs = [float(d) for d in dxs]
    a = axis % ndim
    dxa = dxs[a]
    vol = float(np.prod(dxs))
    trans_sp = vol / dxa  # product of all transverse spacings

    # 1. Global norm
    global_norm = np.sum(np.abs(psi) ** 2) * vol

    # 2. Reduced density matrix & purity (full domain)
    # move active axis to front, flatten transverse into one index
    p = np.moveaxis(psi, a, 0)  # (n_a, *transverse)
    flat = p.reshape(p.shape[0], -1)  # (n_a, n_trans)
    rho = (flat @ np.conj(flat).T) * trans_sp  # (n_a, n_a)
    trace_rho = np.trace(rho) * dxa

    if trace_rho > 1e-12:
        rho_norm = rho / trace_rho
        reduced_purity_snap = np.sum(np.abs(rho_norm) ** 2) * (dxa**2)
    else:
        rho_norm = rho * 0
        reduced_purity_snap = 0.0

    # 3. Reduced purity from Wigner
    norm_factor = np.sum(W) * dxa * dp
    reduced_purity_wig = (
        2 * np.pi * np.sum((W / norm_factor) ** 2) * dxa * dp if norm_factor > 1e-12 else 0.0
    )

    # 4. Von Neumann entanglement entropy
    entanglement_entropy = 0.0
    if trace_rho > 1e-12:
        eigenvalues = np.linalg.eigvalsh(rho_norm) * dxa
        valid_eigs = eigenvalues[eigenvalues > 1e-12]
        entanglement_entropy = -np.sum(valid_eigs * np.log(valid_eigs))

    return global_norm, reduced_purity_snap, reduced_purity_wig, entanglement_entropy
