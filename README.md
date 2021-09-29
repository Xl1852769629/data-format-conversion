# data-format-conversion
### bdd2coco代码使用：

```
Dataset().Bdd2coco(
	bdd100k_json='', # bdd100k json 根路径
	coco_json='' # 保存coco json 根路径
).bdd2coco_train() / .bdd2coco_val() # 训练 / 验证
```

![image-20210929093816750](C:\Users\86185\AppData\Roaming\Typora\typora-user-images\image-20210929093816750.png)

### coco2yolo代码使用：

```
Dataset().Coco2Yolo(
        coco_json='', # input: coco format(json)
        yolo_txt='' # specify where to save the output dir of labels
    ).coco_to_yolo() # 验证集、训练集使用同一个

```

![image-20210929100455848](C:\Users\86185\AppData\Roaming\Typora\typora-user-images\image-20210929100455848.png)

### yolo2coco代码使用：

```
Dataset().Yolo2Coco(
        yolo_txt='', # root path of images and labels, include ./images and ./labels and classes.txt
        coco_json='' # 保存后的coco名称，比如：val.json、train.json
    ).yolo_to_coco()
```

![image-20210929105534131](C:\Users\86185\AppData\Roaming\Typora\typora-user-images\image-20210929105534131.png)

#### 文件夹要求：

**文件夹下要有 images（图片数据），labels（yolo_txt格式），classes（类别信息）：**

**手动生成classes.txt，文件夹命名要求如下：**

![image-20210929104452706](C:\Users\86185\AppData\Roaming\Typora\typora-user-images\image-20210929104452706.png)



**运行代码后会在当前目录生成 annotations 文件夹：**

![image-20210929105802959](C:\Users\86185\AppData\Roaming\Typora\typora-user-images\image-20210929105802959.png)



**annotations 文件夹会有相应的coco_json：**

![image-20210929105917067](C:\Users\86185\AppData\Roaming\Typora\typora-user-images\image-20210929105917067.png)











































