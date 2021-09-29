import os
import cv2
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse


class Dataset(object):
    class Bdd2coco(object):
        def __init__(self, bdd100k_json, coco_json):
            super().__init__()
            self.bdd100k = bdd100k_json
            self.coco = coco_json

        def parse_arguments(self):
            parser = argparse.ArgumentParser(description='BDD100K to COCO format')
            parser.add_argument(
                "-l", "--label_dir",
                # bdd100的json文件
                default=self.bdd100k,
                help="root directory of BDD label Json files",
            )
            parser.add_argument(
                "-s", "--save_path",
                # 保存coco json路径
                default=self.coco,
                help="path to save coco formatted label file",
            )
            return parser.parse_args()

        def bdd2coco(self):
            self.args = self.parse_arguments()

            self.attr_dict = dict()
            self.attr_dict["categories"] = [
                {"supercategory": "none", "id": 1, "name": "person"},
                {"supercategory": "none", "id": 2, "name": "rider"},
                {"supercategory": "none", "id": 3, "name": "car"},
                {"supercategory": "none", "id": 4, "name": "bus"},
                {"supercategory": "none", "id": 5, "name": "truck"},
                {"supercategory": "none", "id": 6, "name": "bike"},
                {"supercategory": "none", "id": 7, "name": "motor"},
                {"supercategory": "none", "id": 8, "name": "traffic light"},
                {"supercategory": "none", "id": 9, "name": "traffic sign"},
                {"supercategory": "none", "id": 10, "name": "train"}
            ]

            self.attr_id_dict = {i['name']: i['id'] for i in self.attr_dict['categories']}
            return self.args, self.attr_dict, self.attr_id_dict

        def bdd2coco_trian(self):
            args, attr_dict, attr_id_dict = self.bdd2coco()
            # ===================================================================train
            # create BDD training set detections in COCO format
            print('Loading training set...')
            with open(os.path.join(args.label_dir,
                                   'bdd100k_labels_images_train.json')) as f:
                train_labels = json.load(f)
            print('Converting training set...')

            out_fn = os.path.join(args.save_path,
                                  'bdd100k_labels_images_det_coco_train.json')
            self.bdd2coco_detection(self.attr_id_dict, train_labels, out_fn)

        def bdd2coco_val(self):
            args, attr_dict, attr_id_dict = self.bdd2coco()
            # ========================================================   val
            print('Loading validation set...')
            # create BDD validation set detections in COCO format
            with open(os.path.join(args.label_dir,
                                   'bdd100k_labels_images_val.json')) as f:
                val_labels = json.load(f)
            print('Converting validation set...')

            out_fn = os.path.join(args.save_path,
                                  'bdd100k_labels_images_det_coco_val.json')
            self.bdd2coco_detection(attr_id_dict, val_labels, out_fn)

        def bdd2coco_detection(self, id_dict, labeled_images, fn):

            images = list()
            annotations = list()

            counter = 0
            for i in tqdm(labeled_images):
                counter += 1
                image = dict()
                image['file_name'] = i['name']
                image['height'] = 720
                image['width'] = 1280

                image['id'] = counter

                empty_image = True

                for label in i['labels']:
                    annotation = dict()
                    if label['category'] in id_dict.keys():
                        empty_image = False
                        annotation["iscrowd"] = 0
                        annotation["image_id"] = image['id']
                        x1 = label['box2d']['x1']
                        y1 = label['box2d']['y1']
                        x2 = label['box2d']['x2']
                        y2 = label['box2d']['y2']
                        annotation['bbox'] = [x1, y1, x2 - x1, y2 - y1]
                        annotation['area'] = float((x2 - x1) * (y2 - y1))
                        annotation['category_id'] = id_dict[label['category']]
                        annotation['ignore'] = 0
                        annotation['id'] = label['id']
                        annotation['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                        annotations.append(annotation)

                if empty_image:
                    continue

                images.append(image)

            self.attr_dict["images"] = images
            self.attr_dict["annotations"] = annotations
            self.attr_dict["type"] = "instances"

            print('saving...')
            json_string = json.dumps(self.attr_dict)
            with open(fn, "w") as file:
                file.write(json_string)

    class Coco2Yolo(object):
        def __init__(self, coco_json, yolo_txt):
            super().__init__()
            self.coco = coco_json
            self.yolo = yolo_txt

        def convert(self, size, box):
            dw = 1. / (size[0])
            dh = 1. / (size[1])
            x = box[0] + box[2] / 2.0
            y = box[1] + box[3] / 2.0
            w = box[2]
            h = box[3]

            x = x * dw
            w = w * dw
            y = y * dh
            h = h * dh
            return (x, y, w, h)

        def coco_yolo_path(self):
            parser = argparse.ArgumentParser()
            parser.add_argument('--json_path',
                                default=self.coco,
                                type=str, help="input: coco format(json)")
            parser.add_argument('--save_path', default=self.yolo, type=str,
                                help="specify where to save the output dir of labels")
            arg = parser.parse_args()
            return arg

        def coco_to_yolo(self):
            arg = self.coco_yolo_path()
            json_file = arg.json_path  # COCO Object Instance 类型的标注
            ana_txt_save_path = arg.save_path  # 保存的路径

            data = json.load(open(json_file, 'r'))
            if not os.path.exists(ana_txt_save_path):
                os.makedirs(ana_txt_save_path)

            id_map = {}  # coco数据集的id不连续！重新映射一下再输出！
            with open(os.path.join(ana_txt_save_path, 'classes.txt'), 'w') as f:
                # 写入classes.txt
                for i, category in enumerate(data['categories']):
                    f.write(f"{category['name']}\n")
                    id_map[category['id']] = i
            # print(id_map)

            for img in tqdm(data['images']):
                filename = img["file_name"]
                img_width = img["width"]
                img_height = img["height"]
                img_id = img["id"]
                head, tail = os.path.splitext(filename)
                ana_txt_name = head + ".txt"  # 对应的txt名字，与jpg一致
                f_txt = open(os.path.join(ana_txt_save_path, ana_txt_name), 'w')
                for ann in data['annotations']:
                    if ann['image_id'] == img_id:
                        box = self.convert((img_width, img_height), ann["bbox"])
                        f_txt.write("%s %s %s %s %s\n" % (id_map[ann["category_id"]], box[0], box[1], box[2], box[3]))
                f_txt.close()

    class Yolo2Coco(object):

        def __init__(self, yolo_txt, coco_json):
            super().__init__()
            self.yolo = yolo_txt
            self.coco = coco_json

        def dataset_path(self):
            parser = argparse.ArgumentParser()

            parser.add_argument('--root_dir', default=self.yolo, type=str,
                                help="root path of images and labels, include ./images and ./labels and classes.txt")
            parser.add_argument('--save_path', type=str, default='./' + self.coco,
                                help="if not split the dataset, give a path to a json file")
            parser.add_argument('--random_split', action='store_true',
                                help="random split the dataset, default ratio is 8:1:1")

            arg = parser.parse_args()
            return arg

        def train_test_val_split(self, img_paths, ratio_train=0.8, ratio_test=0.1, ratio_val=0.1):
            # 这里可以修改数据集划分的比例。
            assert int(ratio_train + ratio_test + ratio_val) == 1
            train_img, middle_img = train_test_split(img_paths, test_size=1 - ratio_train, random_state=233)
            ratio = ratio_val / (1 - ratio_train)
            val_img, test_img = train_test_split(middle_img, test_size=ratio, random_state=233)
            print("NUMS of train:val:test = {}:{}:{}".format(len(train_img), len(val_img), len(test_img)))
            return train_img, val_img, test_img

        def yolo2coco(self, root_path, random_split):
            arg = self.dataset_path()
            originLabelsDir = os.path.join(root_path, 'labels')
            originImagesDir = os.path.join(root_path, 'images')
            with open(os.path.join(root_path, 'classes.txt')) as f:
                classes = f.read().strip().split()
            # images dir name
            indexes = os.listdir(originImagesDir)

            if random_split:
                # 用于保存所有数据的图片信息和标注信息
                train_dataset = {'categories': [], 'annotations': [], 'images': []}
                val_dataset = {'categories': [], 'annotations': [], 'images': []}
                test_dataset = {'categories': [], 'annotations': [], 'images': []}

                # 建立类别标签和数字id的对应关系, 类别id从0开始。
                for i, cls in enumerate(classes, 0):
                    train_dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})
                    val_dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})
                    test_dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})
                train_img, val_img, test_img = self.train_test_val_split(indexes, 0.8, 0.1, 0.1)
            else:
                dataset = {'categories': [], 'annotations': [], 'images': []}
                for i, cls in enumerate(classes, 0):
                    dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})

            # 标注的id
            ann_id_cnt = 0
            for k, index in enumerate(tqdm(indexes)):
                # 支持 png jpg 格式的图片。
                txtFile = index.replace('images', 'txt').replace('.jpg', '.txt').replace('.png', '.txt')
                # 读取图像的宽和高
                im = cv2.imread(os.path.join(root_path, 'images/') + index)
                height, width, _ = im.shape
                if random_split:
                    # 切换dataset的引用对象，从而划分数据集
                    if index in train_img:
                        dataset = train_dataset
                    elif index in val_img:
                        dataset = val_dataset
                    elif index in test_img:
                        dataset = test_dataset
                # 添加图像的信息
                dataset['images'].append({'file_name': index,
                                          'id': k,
                                          'width': width,
                                          'height': height})
                if not os.path.exists(os.path.join(originLabelsDir, txtFile)):
                    # 如没标签，跳过，只保留图片信息。
                    continue
                with open(os.path.join(originLabelsDir, txtFile), 'r') as fr:
                    labelList = fr.readlines()
                    for label in labelList:
                        label = label.strip().split()
                        x = float(label[1])
                        y = float(label[2])
                        w = float(label[3])
                        h = float(label[4])

                        # convert x,y,w,h to x1,y1,x2,y2
                        H, W, _ = im.shape
                        x1 = (x - w / 2) * W
                        y1 = (y - h / 2) * H
                        x2 = (x + w / 2) * W
                        y2 = (y + h / 2) * H
                        # 标签序号从0开始计算, coco2017数据集标号混乱，不管它了。
                        cls_id = int(label[0])
                        width = max(0, x2 - x1)
                        height = max(0, y2 - y1)
                        dataset['annotations'].append({
                            'area': width * height,
                            'bbox': [x1, y1, width, height],
                            'category_id': cls_id,
                            'id': ann_id_cnt,
                            'image_id': k,
                            'iscrowd': 0,
                            # mask, 矩形是从左上角点按顺时针的四个顶点
                            'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
                        })
                        ann_id_cnt += 1

            # 保存结果
            folder = os.path.join(root_path, 'annotations')
            if not os.path.exists(folder):
                os.makedirs(folder)
            if random_split:
                for phase in ['train', 'val', 'test']:
                    json_name = os.path.join(root_path, 'annotations/{}.json'.format(phase))
                    with open(json_name, 'w') as f:
                        if phase == 'train':
                            json.dump(train_dataset, f)
                        elif phase == 'val':
                            json.dump(val_dataset, f)
                        elif phase == 'test':
                            json.dump(test_dataset, f)
                    print('Save annotation to {}'.format(json_name))
            else:
                json_name = os.path.join(root_path, 'annotations/{}'.format(arg.save_path))
                with open(json_name, 'w') as f:
                    json.dump(dataset, f)
                    print('Save annotation to {}'.format(json_name))

        def yolo_to_coco(self):
            arg = self.dataset_path()
            root_path = arg.root_dir
            assert os.path.exists(root_path)
            random_split = arg.random_split
            print("Loading data from ", root_path, "\nWhether to split the data:", random_split)
            self.yolo2coco(root_path, random_split)

# if __name__ == '__main__':
# Dataset().Bdd2coco(
#         bdd100k_json='F:\\dataset\\BDD100K\\bdd100k\\labels\\bdd_one_json\\val',
#         coco_json='F:\\dataset\\BDD100K\\bdd100k\\labels\\coco_json_val').bdd2coco_val()
# Dataset().Coco2Yolo(
#     coco_json='F:\\dataset\\COCO 2017\\annotations\\annotations_trainval2017\\annotations\\instances_val2017.json',
#     yolo_txt='F:\\dataset\\COCO 2017\\annotations\\annotations_trainval2017\\yolo_val_txt'
# ).coco_to_yolo()
# Dataset().Yolo2Coco(
#     yolo_txt='F:\\dataset\\BDD100K\\bdd100k\\labels\\coco_json',
#     coco_json='val.json'
# ).yolo_to_coco()
