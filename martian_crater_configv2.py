import os

custom_imports = dict(imports=["geospatial_fm"])
experiment = "training-output"
project_dir = "/scratch/sp6559/xie-training"
work_dir = os.path.join(project_dir, experiment)
# save_path = work_dir

# base options
dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
auto_resume = True
resume_from = None
cudnn_benchmark = True

epochs = 100
eval_epoch_interval = 5

LEARNING_RATE = 1.5e-5#1.3e-05
LR_CONFIG = dict(
    policy="poly",
    warmup="linear",
    warmup_iters= 1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False,
)
optimizer = dict(type="AdamW", lr=LEARNING_RATE, weight_decay = 0.05, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)
lr_config = LR_CONFIG

log_config = dict(
    interval=5,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=True),
        dict(type='WandbLoggerHook', by_epoch=True, # The Wandb logger is also supported, It requires `wandb` to be installed.
             init_kwargs={'entity': "safipatel", # The entity used to log on Wandb
                          'project': "martian-encoders", # Project name in WandB
                          },
             with_step=False,
        #    log_checkpoint=False,
        #    log_checkpoint_metadata=True,
            ),

    ],
)
checkpoint_config = dict(by_epoch=True, interval=10)
evaluation = dict(
    interval=eval_epoch_interval,
    metric="mIoU",
    pre_eval=True,
    by_epoch=True,
)

ce_weights = [0.2, 0.8] #TODO, NOTE :might need to tune this more
loss_func = loss_decode=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=1,
            class_weight=ce_weights,
            avg_non_ignore=True,
        )#dict(type="DiceLoss", use_sigmoid=False, loss_weight=1, ignore_index=-1)


runner = dict(type="EpochBasedRunner", max_epochs=epochs)
workflow = [("train", 1), ("val", 1)]
norm_cfg = dict(type="BN", requires_grad=True)
FREEZE_BACKBONE = False


# Download burnscars tiff and open them up, see if they are different from the numbers im used to, and if so what means for the means
# burnscars tiff are prenormalized i think
# TODO: change these to the regular Pritvi means/stds that i was using in reconstruction results
img_norm_cfg = dict(
    means=[
        775.2290211032589,
        1080.992780391705,
        1228.5855250417867,
        2497.2022620507532,
        2204.2139147975554,
        1610.8324823273745
    ],
    stds=[
        1281.526139861424,
        1270.0297974547493,
        1399.4802505642526,
        1368.3446143747644,
        1291.6764008585435,
        1154.505683480695,
    ],
)  # change the mean and std of all the bands

num_frames = 1
img_size = 224
num_workers = 4
samples_per_gpu = 4


bands = [0, 1, 2]
tile_size = 224
orig_nsize = 512
crop_size = (tile_size, tile_size)
img_suffix = ".jpg"
seg_map_suffix = "_gtmask.png"
ignore_index = 2
image_nodata = -9999
image_nodata_replace = 0
image_to_float32 = True


num_layers = 12
patch_size = 16
embed_dim = 768
num_heads = 12
tubelet_size = 1
output_embed_dim = num_frames * embed_dim

dataset_type = "RGBDataset" #NOTE: added RGBDataset.py to geospatialfm so that it gets imported AND added it to __all__, this also contains all the new transforms i added
# TO BE DEFINED BY USER: data directory
data_root = os.path.join(project_dir, "dataset")
# model
# TO BE DEFINED BY USER: model path
pretrained_weights_path = os.path.join(project_dir,"prithvi","Prithvi_100M.pt")

train_pipeline = [
    dict(type="LoadRBGImage", **img_norm_cfg, to_float32=image_to_float32), #TODO: change this and the one below to instead load from jpg and multiply to be like HLS data. 
    dict(type="LoadAnnotations", reduce_zero_label=False), #TODO make sure you're loading annotations in the same way as these guys 
    dict(type="BandsExtract", bands=bands),
    dict(type="RandomFlip", prob=0.5),
    dict(type="ToTensor", keys=["img", "gt_semantic_seg"]),
    # to channels first
    dict(type="TorchPermute", keys=["img"], order=(2, 0, 1)), #TODO: check the order of the axis, even in loadgeosptailaimage. But this should be the same, realisticalyl
    dict(type="TorchNormalizeAndDuplicate", **img_norm_cfg, duplicate = True),
    dict(type="TorchRandomCrop", crop_size=(tile_size, tile_size)),
    dict(
        type="Reshape",
        keys=["img"],
        new_shape=(6, num_frames, tile_size, tile_size),
    ),
    dict(type="Reshape", keys=["gt_semantic_seg"], new_shape=(1, tile_size, tile_size)),
    dict(type="CastTensor", keys=["gt_semantic_seg"], new_type="torch.LongTensor"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]
test_pipeline = [
    dict(type="LoadRBGImage", **img_norm_cfg, to_float32=image_to_float32),
    dict(type="BandsExtract", bands=bands),
    dict(type="ToTensor", keys=["img"]),
    # to channels first
    dict(type="TorchPermute", keys=["img"], order=(2, 0, 1)),
    dict(type="TorchNormalizeAndDuplicate", **img_norm_cfg, duplicate = True),
    dict(
        type="Reshape",
        keys=["img"],
        new_shape=(6, num_frames, -1, -1),
        look_up=dict({"2": 1, "3": 2}),
    ),
    dict(type="CastTensor", keys=["img"], new_type="torch.FloatTensor"),
    dict(
        type="CollectTestList",
        keys=["img"],
        meta_keys=[
            "img_info",
            "seg_fields",
            "img_prefix",
            "seg_prefix",
            "filename",
            "ori_filename",
            "img",
            "img_shape",
            "ori_shape",
            "pad_shape",
            "scale_factor",
            "img_norm_cfg",
        ],
    ),
]
num_classes = 2
CLASSES = ("Not within crater", "Within crater")

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=num_workers,
    train=dict(
        type=dataset_type,
        CLASSES=CLASSES,
        data_root=data_root,
        img_dir="images/train",
        ann_dir="annotations/train",
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        pipeline=train_pipeline,
        ignore_index=ignore_index,
    ),
    val=dict(
        type=dataset_type,
        CLASSES=CLASSES,
        data_root=data_root,
        img_dir="images/valid",
        ann_dir="annotations/valid",
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        pipeline=test_pipeline,
        ignore_index=ignore_index,
    ),
    test=dict(
        type=dataset_type,
        CLASSES=CLASSES,
        data_root=data_root,
        img_dir="images/test",
        ann_dir="annotations/test",
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        pipeline=test_pipeline,
        ignore_index=ignore_index,
    ),
)




model = dict(
    type="TemporalEncoderDecoder",
    frozen_backbone=FREEZE_BACKBONE,
    backbone=dict(
        type="TemporalViTEncoder",
        pretrained=pretrained_weights_path,
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        in_chans=6,
        embed_dim=embed_dim,
        depth=12,
        num_heads=num_heads,
        mlp_ratio=4.0,
        norm_pix_loss=False,
    ),
    neck=dict(
        type="ConvTransformerTokensToEmbeddingNeck",
        embed_dim=embed_dim * num_frames,
        output_embed_dim=output_embed_dim,
        drop_cls_token=True,
        Hp=14,
        Wp=14,
    ),
    decode_head=dict(
        num_classes=len(CLASSES),
        in_channels=output_embed_dim,
        type="FCNHead",
        in_index=-1,
        ignore_index=ignore_index,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=loss_func,
    ),
    auxiliary_head=dict(
        num_classes=len(CLASSES),
        in_channels=output_embed_dim,
        type="FCNHead",
        in_index=-1,
        channels=256,
        num_convs=2,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=dict(type="BN", requires_grad=True),
        align_corners=False,
        loss_decode=loss_func,
    ),
    train_cfg=dict(),
    test_cfg=dict(
        mode="slide",
        stride=(int(tile_size / 2), int(tile_size / 2)),
        crop_size=(tile_size, tile_size),
    ),
)
