"""Smoke tests that need no dataset, no checkpoint and no GPU.

They check that:
  * every Python file compiles,
  * the model modules import and give the expected output size on random input,
  * the degradation and normalisation helpers give sane shapes and ranges,
  * a few documented facts (parameter count, curriculum, constants that must
    stay in sync) still match the code.

They do NOT train, download data or test the scripts that are known to be
broken in the committed code (see docs/method.md, "Known issues").
"""

import importlib
import py_compile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

# conftest.py puts src/ on sys.path.
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

# SCUNet settings used by scunet_train.py, evaluate.py, inference.py and
# quick_benchmark.py.
SCUNET_KWARGS = dict(
    in_channels=1,
    out_channels=1,
    img_size=64,
    nc=96,
    depth=(2, 2, 2, 2),
    num_heads=6,
    window_size=8,
    mlp_ratio=4.0,
    drop_path_rate=0.0,
    scale=2,
    residual_output=True,
)


def _count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── 1. Every file compiles ──────────────────────────────────────────────────

PY_FILES = sorted(SRC.glob("*.py")) + [ROOT / "approch_other.py"]


@pytest.mark.parametrize("path", PY_FILES, ids=lambda p: p.name)
def test_python_file_compiles(path):
    py_compile.compile(str(path), doraise=True)


# ── 2. Modules without side effects import cleanly ──────────────────────────
# Scripts such as download_*.py or explore_dataset.py run network code at
# import time, so they are only compiled (test 1), never imported.

IMPORTABLE = [
    "sar_dataset",
    "scunet_model",
    "swinIR_model",
    "model",
    "scunet_train",
    "inference",
    "evaluate",
    "quick_benchmark",
    "diagnose_scunet_histogram",
]


@pytest.mark.parametrize("name", IMPORTABLE)
def test_module_imports(name):
    importlib.import_module(name)


# ── 3. Models: forward pass on random input ─────────────────────────────────

def test_scunet_forward_gives_2x_output():
    from scunet_model import SCUNet

    torch.manual_seed(0)
    model = SCUNet(**SCUNET_KWARGS).eval()
    x = torch.rand(2, 1, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 1, 128, 128)
    assert torch.isfinite(y).all()


def test_scunet_residual_mode_starts_as_bicubic():
    """In residual mode the last conv starts at zero, so sr == bicubic(lr)."""
    from scunet_model import SCUNet

    torch.manual_seed(0)
    model = SCUNet(**SCUNET_KWARGS).eval()
    x = torch.rand(1, 1, 64, 64)
    with torch.no_grad():
        residual = model(x)
        sr = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False) + residual
        bicubic = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)
    assert torch.count_nonzero(residual) == 0
    assert torch.equal(sr, bicubic)


def test_scunet_parameter_count_matches_readme():
    from scunet_model import SCUNet

    assert _count_params(SCUNet(**SCUNET_KWARGS)) == 32_952_907


def test_swinir_forward_gives_4x_output():
    from swinIR_model import SwinIR

    torch.manual_seed(0)
    model = SwinIR(
        in_channels=1, out_channels=1, img_size=64, embed_dim=60,
        depths=(6, 6, 6, 6), num_heads=(6, 6, 6, 6), window_size=8,
        mlp_ratio=2.0, scale=4, drop_path_rate=0.1,
    ).eval()
    x = torch.rand(1, 1, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 256, 256)
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0  # sigmoid output
    assert _count_params(model) == 1_195_261


@pytest.mark.parametrize("num_blocks, expected_params", [(8, 7_896_129), (23, 22_006_689)])
def test_rrdbnet_forward_gives_4x_output(num_blocks, expected_params):
    from model import RRDBNet

    torch.manual_seed(0)
    model = RRDBNet(in_channels=1, out_channels=1, num_feat=64,
                    num_blocks=num_blocks, growth_rate=32, scale=4).eval()
    assert _count_params(model) == expected_params
    x = torch.rand(1, 1, 32, 32)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 128, 128)
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0  # sigmoid output


# ── 4. Degradation and normalisation (no GeoTIFF needed) ────────────────────

def _fake_sar_patch(size=128, seed=0):
    """Random uint16 'amplitude' patch with a speckle-like gamma distribution."""
    rng = np.random.default_rng(seed)
    return (rng.gamma(shape=2.0, scale=300.0, size=(size, size))).astype(np.uint16)


def test_degrade_halves_the_patch_size():
    import sar_dataset

    np.random.seed(0)
    hr = _fake_sar_patch(128)
    lr = sar_dataset.degrade(hr)
    assert sar_dataset.SCALE_FACTOR == 2
    assert lr.shape == (64, 64)
    assert lr.dtype == np.float32
    assert np.isfinite(lr).all()
    assert lr.min() >= 0.0


def test_normalize_joint_keeps_values_in_unit_range():
    import sar_dataset

    np.random.seed(0)
    hr_raw = _fake_sar_patch(128)
    lr_raw = sar_dataset.degrade(hr_raw)
    hr, lr = sar_dataset.normalize_joint(hr_raw, lr_raw)
    assert hr.shape == (128, 128) and lr.shape == (64, 64)
    for arr in (hr, lr):
        assert arr.dtype == np.float32
        assert arr.min() >= 0.0 and arr.max() <= 1.0


def test_inference_degradation_constants_match_sar_dataset():
    """inference.py keeps its own copy of the constants; they must stay in sync."""
    import inference
    import sar_dataset

    assert inference.SCALE == sar_dataset.SCALE_FACTOR
    assert inference.PSF_SIGMA == sar_dataset.PSF_SIGMA
    assert inference.SPECKLE_LOOKS == sar_dataset.SPECKLE_LOOKS
    assert inference.NOISE_STD == sar_dataset.NOISE_STD
    assert inference.DEGRADE_BLEND == sar_dataset.DEGRADE_BLEND
    assert inference.LR_TILE * inference.SCALE == sar_dataset.PATCH_SIZE


def test_inference_helpers_pad_and_round_trip():
    import inference

    scene = _fake_sar_patch(100, seed=1)
    padded, pad_h, pad_w = inference.pad_to_multiple(scene.astype(np.float32), inference.LR_TILE)
    assert padded.shape == (128, 128) and (pad_h, pad_w) == (28, 28)

    norm, p1, p99 = inference.normalize_scene(scene)
    assert norm.min() >= 0.0 and norm.max() <= 1.0
    back = inference.denormalize_scene(norm, p1, p99)
    assert back.dtype == np.uint16 and back.shape == scene.shape


# ── 5. Training curriculum (documented in README and docs/method.md) ────────

def test_degradation_curriculum_matches_docs():
    import scunet_train

    first, last = scunet_train.DEGRADE_CURRICULUM[0], scunet_train.DEGRADE_CURRICULUM[-1]
    assert first == (1, 0.22, 28, 0.00030, 0.65)
    assert last == (71, 0.10, 96, 0.00005, 0.25)
    assert scunet_train.CFG["curriculum_interp"] == "linear"

    assert scunet_train.get_curriculum_stage(1) == first
    assert scunet_train.get_curriculum_stage(100) == last

    # Epoch 11 is half-way between the epoch-1 and epoch-21 anchors.
    _, sigma, looks, noise, blend = scunet_train.get_curriculum_stage(11)
    assert sigma == pytest.approx(0.20)
    assert looks == 34
    assert noise == pytest.approx(0.00025)
    assert blend == pytest.approx(0.575)


# ── 6. Known-broken scripts: documented, not tested as passing ──────────────

@pytest.mark.skip(reason=(
    "Known issue: train.py (RRDBNet, 4x, 256 px HR) expects 4x degradation, but "
    "sar_dataset.SCALE_FACTOR is 2, so LR is 128 px and the model outputs 512 px "
    "against a 256 px target. See docs/method.md, 'Known issues'."
))
def test_train_rrdb_shapes_match_dataset():
    pass


@pytest.mark.skip(reason=(
    "Known issue: swinIR_train.py builds SwinIR for a 64 px LR input (4x), but with "
    "sar_dataset.SCALE_FACTOR = 2 the LR patch is 128 px and the forward pass fails. "
    "See docs/method.md, 'Known issues'."
))
def test_train_swinir_shapes_match_dataset():
    pass
