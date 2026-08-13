# Third-party licences

This project is MIT (see `LICENSE`), but it builds on and redistributes the
following third-party components. Each licence below must travel with any
redistribution; the relevant notice is reproduced in full where required.

| Component | Licence | Used for | Source |
|---|---|---|---|
| Community-Forensics (`OwensLab/commfor-model-224`, `-384`) | MIT | The two detection models, converted to fp16 ONNX and redistributed in the built extension | https://github.com/JeongsooP/Community-Forensics (Park & Owens, CVPR 2025) — notice reproduced in `extension/models/LICENSE-community-forensics.txt` |
| onnxruntime-web | MIT (Microsoft) | Browser inference engine, vendored into `extension/lib/` by `build.py` | https://github.com/microsoft/onnxruntime |
| AI Detector Arena Benchmark v0.1 | CC-BY-4.0 (data) | The 1,200-image proxy evaluation set the harness measures against | https://aidetectarena.com/datasets/v0.1 |
| Unsplash photos | Unsplash License | The "real" class of the evaluation set and browser smoke-test images | https://unsplash.com/license |

## Community-Forensics (MIT)

Reproduced in full at `extension/models/LICENSE-community-forensics.txt`,
which sits next to the two converted model files it applies to.

- Copyright (c) 2025 Jeongsoo Park
- Source: https://github.com/JeongsooP/Community-Forensics

## onnxruntime-web (MIT, Microsoft)

The onnxruntime-web distribution (including the WebGPU/WebGL/WASM kernels in
`extension/lib/`) is copyright Microsoft and licensed under the MIT licence,
an MIT notice, and the EULA for binary distributions. Full licence text and
third-party notices: https://github.com/microsoft/onnxruntime/blob/main/LICENSE

## AI Detector Arena Benchmark v0.1 (CC-BY-4.0)

The benchmark dataset asset used to build the proxy evaluation set is
licensed under Creative Commons Attribution 4.0 International. Attribution:
AI Detector Arena. Source: https://aidetectarena.com/datasets/v0.1. Full text:
https://creativecommons.org/licenses/by/4.0/

## Unsplash photos (Unsplash License)

Real photographs in the evaluation set and smoke tests are from Unsplash,
licensed under the Unsplash License, which permits free use (commercial and
non-commercial) without asking permission or providing attribution, with
certain restrictions (no redistribution of the photos in aggregate as a stock
photo service, no claiming them as your own). Source: https://unsplash.com/license