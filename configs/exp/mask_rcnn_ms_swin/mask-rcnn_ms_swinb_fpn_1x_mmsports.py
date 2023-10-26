_base_ = [
    '../../_base_/models/mask-rcnn_r50_fpn.py',
    '../../_base_/datasets/mmsports_instance.py',
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py'
]

model = dict(
    type='MaskScoringRCNN',
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=False,
        frozen_stages=-1,
        init_cfg=None),
    neck=dict(in_channels=[128, 256, 512, 1024]),
    roi_head = dict(
        type='MaskScoringRoIHead',
        bbox_head = dict(num_classes=1),
        mask_head = dict(num_classes=1),
        mask_iou_head=dict(
            type='MaskIoUHead',
            num_convs=4,
            num_fcs=2,
            roi_feat_size=14,
            in_channels=256,
            conv_out_channels=256,
            fc_out_channels=1024,
            num_classes=1)
    ),
    train_cfg=dict(rcnn=dict(mask_thr_binary=0.5))
)

train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)