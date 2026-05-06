import torch
ckpt = torch.load(r'd:\kun-data\kun-code-data\反无\cpp智能控制\data\train\dscgnet_legacy_control_best.pt', map_location='cpu')
for k, v in ckpt['model_state_dict'].items():
    print(k, v.shape)
print('---')
print(ckpt.get('experiment', ))
