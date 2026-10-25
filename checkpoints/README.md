Put warm-start checkpoints here, e.g. `vitunetr_liver_2class.pt` saved from the ViT-UNETR notebook with

```python
torch.save(model.state_dict(), "vitunetr_liver_2class.pt")
```

and set `segmenter.init_from: checkpoints/vitunetr_liver_2class.pt` in the config.
