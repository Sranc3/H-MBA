_base_ = [
    'DEFAULT_TRAIN_GQA_VARIANT.py',
    'DEFAULT_TRAIN_CLEVR_VARIANT.py',
    'DEFAULT_TRAIN_POINT_VARIANT.py',
    'DEFAULT_TRAIN_GPTGEN_VARIANT.py',
    'DEFAULT_TRAIN_VCR_VARIANT.py',
    'DEFAULT_TRAIN_VQAv2_VARIANT.py',
    'DEFAULT_TRAIN_VQAEX_VARIANT.py',
    #'DEFAULT_TRAIN_DRAMA.py',
]

DEFAULT_TRAIN_DATASET = dict(
    flickr=dict(
        type='FlickrDataset',
        filename=r'{{fileDirname}}/../../../data/CWB_flickr30k_train.jsonl',
        image_folder=r'zz1424:s3://production-public-flickr_image/Flickr_Image/unzip/flickr30k_images/flickr30k_images',
        template_file=r'{{fileDirname}}/template/flickr30k.json',
    ),
    rec=dict(
        type='RECDataset',
        filename=r'{{fileDirname}}/../../../data/REC_ref3_train.jsonl',
        image_folder=r'zz1424:s3://visual_grounding/academic_data/refer/images/mscoco/images/train2014/',
        template_file=r'{{fileDirname}}/template/REC.json',
    ),
    recvg=dict(
        type='RECDataset',
        filename=r'{{fileDirname}}/../../../data/GC_genome196_train.jsonl',
        image_folder=r'zz1424:s3://publicdataset_8/Visual_Genome_Dataset_V1.2/unzip/data',
        template_file=r'{{fileDirname}}/template/REC.json',
    ),
    reg=dict(
        type='REGDataset',
        filename=r'{{fileDirname}}/../../../data/REC_ref3_train.jsonl',
        image_folder=r'zz1424:s3://visual_grounding/academic_data/refer/images/mscoco/images/train2014/',
        template_file=r'{{fileDirname}}/template/REG.json',
    ),
    gc=dict(
        type='GCDataset',
        filename=r'{{fileDirname}}/../../../data/GC_genome196_train.jsonl',
        image_folder=r'zz1424:s3://publicdataset_8/Visual_Genome_Dataset_V1.2/unzip/data',
        template_file=r'{{fileDirname}}/template/GC.json',
    ),
    dramasc=dict(
        type='FlickrDataset',
        filename=r'/mnt/sdf/json_file/drama_train_SC.jsonl',
        image_folder=r'/mnt/sdf/drama_data/format/train',
        template_file=r'{{fileDirname}}/template/drama.json',
    ),
    caption=dict(
        type='CaptionDataset',
        filename=r'{{fileDirname}}/../../../data/CAP_coco2014_train.jsonl',
        image_folder=r'openmmlab1424:s3://openmmlab/datasets/detection/coco/train2014/',
        template_file=r'{{fileDirname}}/template/image_cap.json',
    ),
    drama=dict(
        type='InstructDataset',
        filename=r"/mnt/sdf/drama_data/drama_train.jsonl",   ####暂时先选这种style吧，没办法了
        image_folder=r'/mnt/sdf/drama_data/format/train',  # TODO: zz make folder name mistake
    ),
    bddx=dict(
        type='InstructDataset',
        filename=r"/home/mydata/BDDX/shikra_json/bddx_train_action.jsonl",   ####替换 justification
        image_folder=r"/home/mydata/BDDX/Images/train",  # TODO: zz make folder name mistake
    ),
    llavacc3m=dict(
        type='InstructDataset',
        filename=r"{{fileDirname}}/../../../data/llava_cc3m.jsonl",   ####暂时先选这种style吧，没办法了
        image_folder=r'sh41:s3://MultiModal/Monolith/academic/llava-pretrain/data/558K_imgs',  # TODO: zz make folder name mistake
    ),
    llavalcs=dict(
        type='InstructDataset',
        filename=r"{{fileDirname}}/../../../data/blip_laion_cc_sbu_558k.jsonl",
        image_folder=r'sh41:s3://MultiModal/Monolith/academic/llava-pretrain/data/595K_imgs',  # TODO: zz make folder name mistake
    ),
    instruct=dict(
        type='InstructDataset',
        filename=r'{{fileDirname}}/../../../data/llava_instruct_150k.jsonl',
        image_folder=r'zz1424:s3://PublicDatalist/public_datalist_6_unzip/train2014',
        add_coco_prefix=True,
    ),
    **_base_.DEFAULT_TRAIN_GQA_VARIANT,
    **_base_.DEFAULT_TRAIN_CLEVR_VARIANT,
    **_base_.DEFAULT_TRAIN_POINT_VARIANT,
    **_base_.DEFAULT_TRAIN_GPTGEN_VARIANT,
    **_base_.DEFAULT_TRAIN_VCR_VARIANT,
    **_base_.DEFAULT_TRAIN_VQAv2_VARIANT,
    **_base_.DEFAULT_TRAIN_VQAEX_VARIANT,
)
