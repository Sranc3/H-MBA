from ..root import DATASETS
from ..utils import MInstrDataset


@DATASETS.register_module()
class InstructDataset(MInstrDataset):
    def __init__(self, *args, add_coco_prefix=False, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(), template_string='', template_file=None)
        self.add_coco_prefix = add_coco_prefix

    def __getitem__(self, index):
        item = self.get_raw_item(index)
        if self.add_coco_prefix:
            img_path = f"COCO_train2014_{item['image']}"
        else:
            img_path = item['image']
        conversations = item['conversations']
        img_path_p1 = f"{item['id']}_p1.jpg"
        img_path_p2 = f"{item['id']}_p2.jpg"
        img_path_p3 = f"{item['id']}_p3.jpg"
        img_path_p4 = f"{item['id']}_p4.jpg"
        img_path_p5 = f"{item['id']}_p5.jpg"
        img_path_p6 = f"{item['id']}_p6.jpg"
        img_path_p7 = f"{item['id']}_p7.jpg"

        img1 = self.get_image(img_path_p1)
        img2 = self.get_image(img_path_p2)
        img3 = self.get_image(img_path_p3)
        img4 = self.get_image(img_path_p4)
        img5 = self.get_image(img_path_p5)
        img6 = self.get_image(img_path_p6)
        img7 = self.get_image(img_path_p7)
        video = [img1,img2,img3,img4,img5,img6,img7]

        image = self.get_image(img_path)
        ret = {
            'image': image,
            'video' : video,
            'conversations': conversations,
        }
        return ret
