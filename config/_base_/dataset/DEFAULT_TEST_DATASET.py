_base_ = [
    'DEFAULT_TEST_REC_VARIANT.py',
    'DEFAULT_TEST_FLICKR_VARIANT.py',
    'DEFAULT_TEST_GQA_VARIANT.py',
    'DEFAULT_TEST_CLEVR_VARIANT.py',
    'DEFAULT_TEST_GPTGEN_VARIANT.py',
    'DEFAULT_TEST_VCR_VARIANT.py',
    'DEFAULT_TEST_VQAv2_VARIANT.py',
    'DEFAULT_TEST_POINT_VARIANT.py',
    'DEFAULT_TEST_POPE_VARIANT.py',
]

DRAMA_TEST_COMMON_CFG = dict(
    type='FlickrDataset',
    max_dynamic_size=None,
)

DEFAULT_TEST_DATASET = dict(
    drama=dict(
        type='InstructDataset',
        filename=r"/mnt/sdf/drama_data/drama_test.jsonl",   ####暂时先选这种style吧，没办法了
        image_folder=r'/mnt/sdf/drama_data/format/test',  # TODO: zz make folder na
    ),
    dramasc=dict(
        **DRAMA_TEST_COMMON_CFG,
        filename=r'/mnt/sdf/json_file/drama_test_SC.jsonl',                  ########################  在这里进行长尾和真实情况的测试
        #filename=r'/mnt/sdf/drama_data/format/ltail/ltail.jsonl',
        image_folder=r'/mnt/sdf/drama_data/format/test',
        #image_folder=r'/mnt/sdf/drama_data/format/ltail',
        template_file=r'{{fileDirname}}/template/drama.json',
    ),
    nuscenes=dict(
        **DRAMA_TEST_COMMON_CFG,
        filename=r'/mnt/sdf/json_file/nuscenes_test1_SC.jsonl', ####应该需要自己生成,测试可以不用？
        image_folder=r'/mnt/sdc/nuScenes',   #####
        template_file=r'{{fileDirname}}/template/nuscenes.json',
    ),
    nuscenes_train=dict(
        **DRAMA_TEST_COMMON_CFG,
        filename=r'/mnt/sdf/json_file/nuscenes_train1_SC.jsonl', ####应该需要自己生成,测试可以不用？
        image_folder=r'/mnt/sdc/nuScenes',   #####
        template_file=r'{{fileDirname}}/template/nuscenes.json',
    ),
    bddx=dict(
        type='InstructDataset',
        filename=r"/home/mydata/BDDX/shikra_json/bddx_test_justification.jsonl",   ####替换 justification vis
        image_folder=r"/home/mydata/BDDX/Images/test",  # TODO: 
        #template_file=r'{{fileDirname}}/template/bddx.json',
    ),
    **_base_.DEFAULT_TEST_REC_VARIANT,
    **_base_.DEFAULT_TEST_FLICKR_VARIANT,
    **_base_.DEFAULT_TEST_GQA_VARIANT,
    **_base_.DEFAULT_TEST_CLEVR_VARIANT,
    **_base_.DEFAULT_TEST_GPTGEN_VARIANT,
    **_base_.DEFAULT_TEST_VCR_VARIANT,
    **_base_.DEFAULT_TEST_VQAv2_VARIANT,
    **_base_.DEFAULT_TEST_POINT_VARIANT,
    **_base_.DEFAULT_TEST_POPE_VARIANT,
)
