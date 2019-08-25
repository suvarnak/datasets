<div itemscope itemtype="http://schema.org/Dataset">
  <div itemscope itemprop="includedInDataCatalog" itemtype="http://schema.org/DataCatalog">
    <meta itemprop="name" content="TensorFlow Datasets" />
  </div>
  <meta itemprop="name" content="cifar10" />
  <meta itemprop="description" content="The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images." />
  <meta itemprop="url" content="https://www.tensorflow.org/datasets/catalog/cifar10" />
  <meta itemprop="sameAs" content="https://www.cs.toronto.edu/~kriz/cifar.html" />
</div>

# `cifar10`

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with
6000 images per class. There are 50000 training images and 10000 test images.

*   URL:
    [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
*   `DatasetBuilder`:
    [`tfds.image.cifar.Cifar10`](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/image/cifar.py)
*   Version: `v1.0.2`
*   Size: `162.17 MiB`

## Features
```python
FeaturesDict({
    'image': Image(shape=(32, 32, 3), dtype=tf.uint8),
    'label': ClassLabel(shape=(), dtype=tf.int64, num_classes=10),
})
```

## Statistics

Split | Examples
:---- | -------:
ALL   | 60,000
TRAIN | 50,000
TEST  | 10,000

## Urls

*   [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)

## Supervised keys (for `as_supervised=True`)
`(u'image', u'label')`

## Citation
```
@TECHREPORT{Krizhevsky09learningmultiple,
    author = {Alex Krizhevsky},
    title = {Learning multiple layers of features from tiny images},
    institution = {},
    year = {2009}
}
```

--------------------------------------------------------------------------------
