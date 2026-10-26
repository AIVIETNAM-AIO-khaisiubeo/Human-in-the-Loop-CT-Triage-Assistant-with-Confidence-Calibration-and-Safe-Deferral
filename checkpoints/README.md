Optional warm-start checkpoints for the segmentation tool.

The ViT-UNETR 2.5D in `radagent/models/torch_models.py` is the same architecture as the
`ViTUNETR25D_LiverSegmentation` notebook, extended from 2 classes (background / liver) to 3
(background / liver / tumour). A liver model saved from the notebook with

```python
torch.save(model.state_dict(), "checkpoints/vitunetr_liver_2class.pt")
```

can be used to start training: set `segmenter.init_from: checkpoints/vitunetr_liver_2class.pt`
in `configs/lits.yaml`. Every layer whose shape matches is copied; the 3-class output head is
newly initialised. Both use the same notebook-compatible case split, so no test case leaks
into training.

Trained run checkpoints live in `outputs/<run>/model/` (not committed - see `.gitignore`).
MedGemma weights are downloaded by Hugging Face into its own cache (`~/.cache/huggingface`).
