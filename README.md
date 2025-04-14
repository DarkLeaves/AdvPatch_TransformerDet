# Adversarial Patch against Transformer-based object detectors

Use Transformer-based detectors as source model to generate adversarial patch.

## Available source model:

* [X] DeformableDETR

## Available evalate model:

* [ ] DETR
* [ ] DeformableDETR
* [ ] SwinTrans

## Based On

This project is based on [T-SEA](https://github.com/VDIGPKU/T-SEA).
Many thanks to the original authors for their open-source contribution.

## Env construct

```bash
conda create -n adv python=3.9 -y
conda activate adv
pip install -r requirements.txt
```

❗️Installing the `pytorchyolo` (a lib used in T-SEA) in "requirements.txt"  will automatically install  `pytorch`=1.12. 

If this version of the torch does not match your GPU, please uninstall the `torch` `torchvision` and install the matching version.
