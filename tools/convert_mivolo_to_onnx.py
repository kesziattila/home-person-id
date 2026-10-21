"""Export MiVOLO v2 (d1_384, face+body age model) to ONNX for `age.model` (desktop, one-off).

The runtime (`src/recognition/age_estimator.py`) needs only the ONNX / TensorRT file; the
export needs the MiVOLO model code and timm 0.8.13.dev0, which are NOT project
dependencies — keep them out of the venv:

    mkdir -p models/mivolo_v2 /tmp/mivolo/mivolo/model /tmp/mivolo/mivolo/data
    curl -L -o models/mivolo_v2/model.safetensors \\
        https://huggingface.co/iitolstykh/mivolo_v2/resolve/main/model.safetensors
    for f in model/create_timm_model.py model/mivolo_model.py model/cross_bottleneck_attn.py data/misc.py; do
        curl -L -o /tmp/mivolo/mivolo/$f https://raw.githubusercontent.com/WildChlamydia/MiVOLO/main/mivolo/$f
    done                                   # read them before running: they are imported below
    pip install --no-deps --target /tmp/mivolo/pylib timm==0.8.13.dev0
    pip install safetensors                # only for this export
    PYTHONPATH=. python tools/convert_mivolo_to_onnx.py --src /tmp/mivolo

VOLO's outlook attention uses F.fold (ONNX Col2Im, opset 18 only, not TensorRT friendly);
it is swapped for an identical grouped conv_transpose2d with one-hot kernels. The script
checks the swap and the ONNX model against PyTorch (2026-10-19: 1e-7 / 1.4e-4).
Then build the TensorRT engine on the Jetson (docs/JETSON.md).
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def fold_as_conv_transpose(x, output_size, kernel_size, padding=0, stride=1, dilation=1):
    """F.fold for (B, C*k*k, L) inputs as a grouped transposed convolution (same result)."""
    import torch
    import torch.nn.functional as F

    k = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
    s = stride if isinstance(stride, int) else stride[0]
    p = padding if isinstance(padding, int) else padding[0]
    batch, ckk, _ = x.shape
    channels = ckk // (k * k)
    out_h, out_w = output_size
    h = (out_h + 2 * p - k) // s + 1
    w = (out_w + 2 * p - k) // s + 1
    weight = torch.eye(k * k, dtype=x.dtype).reshape(k * k, 1, k, k).repeat(channels, 1, 1, 1)
    full_h, full_w = (h - 1) * s - 2 * p + k, (w - 1) * s - 2 * p + k
    return F.conv_transpose2d(x.reshape(batch, ckk, h, w), weight, stride=s, padding=p, groups=channels,
                              output_padding=(out_h - full_h, out_w - full_w))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="dir with mivolo/ sources and pylib/ (timm)")
    parser.add_argument("--weights", default="models/mivolo_v2/model.safetensors")
    parser.add_argument("--out", default="models/mivolo_v2/mivolo_d1_384.onnx")
    args = parser.parse_args()
    sys.path[:0] = [args.src, str(Path(args.src) / "pylib")]

    import onnxruntime as ort
    import torch
    import torch.nn.functional as F
    import timm.models.volo as volo
    from safetensors.torch import load_file
    from mivolo.model.create_timm_model import create_model

    net = create_model("mivolo_d1_384", num_classes=3, in_chans=6, pretrained=False)
    state = {k.removeprefix("mivolo.model."): v.float() for k, v in load_file(args.weights).items()}
    missing, unexpected = net.load_state_dict(state, strict=False)
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    net.eval()

    x = torch.randn(2, 6, 384, 384)
    with torch.no_grad():
        ref = net(x).numpy()
        patched = type("F", (), {k: getattr(F, k) for k in dir(F) if not k.startswith("__")})
        patched.fold = staticmethod(fold_as_conv_transpose)
        volo.F = patched
        swapped = net(x).numpy()
    print(f"F.fold vs conv_transpose2d: max diff {np.abs(ref - swapped).max():.2e}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(net, x[:1], args.out, input_names=["input"], output_names=["output"],
                      opset_version=17, dynamo=False)
    out = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"]).run(None, {"input": x[:1].numpy()})[0]
    print(f"ONNX vs PyTorch: max diff {np.abs(ref[:1] - out).max():.2e} -> {args.out}")


if __name__ == "__main__":
    main()
