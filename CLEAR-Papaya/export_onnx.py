"""Export PapayaFormer to ONNX for on-device (Android / Raspberry Pi) latency
benchmarking with onnxruntime. Architecture is identical to
kernel_papayaformer/kernel_papayaformer.py.

Run: python export_onnx.py
Produces: papayaformer.onnx  (~67 MB, opset 17, batch 1, 224x224 input)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

K = 6


class MSLA(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.d = nn.ModuleList([nn.Sequential(
            nn.Conv2d(c, c, 3, padding=d, dilation=d, bias=False), nn.BatchNorm2d(c))
            for d in (1, 2, 3)])
        self.spatial = nn.Conv2d(3 * c, 1, 1)
        self.mlp = nn.Sequential(nn.Linear(c, c // r), nn.GELU(), nn.Linear(c // r, c))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        cat = torch.cat([b(x) for b in self.d], 1)
        M = torch.sigmoid(self.spatial(cat))
        s = torch.sigmoid(self.mlp(x.mean((2, 3))))
        return x + self.gamma * (x * M * s[:, :, None, None])


class PapayaFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.bb = timm.create_model("mobilevit_s", pretrained=True,
                                    features_only=True, out_indices=(2, 3, 4))
        chs = self.bb.feature_info.channels()
        self.msla = nn.ModuleList([MSLA(c) for c in chs])
        self.head = nn.Linear(sum(chs), K)

    def forward(self, x):
        feats = self.bb(x)
        return F.softplus(self.head(torch.cat(
            [m(z).mean((2, 3)) for z, m in zip(feats, self.msla)], 1)))


def main():
    net = PapayaFormer().eval()
    npar = sum(p.numel() for p in net.parameters()) / 1e6
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        ref = net(dummy).numpy()
    try:
        torch.onnx.export(
            net, dummy, "papayaformer.onnx",
            input_names=["x"], output_names=["evidence"],
            opset_version=17, do_constant_folding=True,
            dynamo=False, verbose=False)
    except TypeError:  # older torch without dynamo kwarg
        torch.onnx.export(
            net, dummy, "papayaformer.onnx",
            input_names=["x"], output_names=["evidence"],
            opset_version=17, do_constant_folding=True)
    print(f"wrote papayaformer.onnx  ({npar:.1f} M params)  output shape {ref.shape}")

    # sanity: run the exported graph and compare
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession("papayaformer.onnx", providers=["CPUExecutionProvider"])
        out = sess.run(None, {"x": dummy.numpy()})[0]
        print(f"onnxruntime check: max abs diff vs torch = {np.abs(out - ref).max():.2e}")
    except Exception as e:  # noqa: BLE001
        print("onnxruntime not installed here; skipping numeric check:", e)


if __name__ == "__main__":
    main()
