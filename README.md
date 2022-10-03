路径及分类存储在csv文件中，图片在一个文件夹中

1、首先将数据集划分为训练集，验证集和测试集，在每个数据集中按照分类对图片进行分别保存在不同文件夹中

2、读取数据集有两种方式，在read_rice.py文件中，构建MyDataSet类，再进行dataloader

第二种方法直接使用

```python
torchvision.datasets.ImageFolder(root=dataset_dir+"train", transform = train_transform)
```

