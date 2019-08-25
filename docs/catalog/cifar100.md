<div itemscope itemtype="http://schema.org/Dataset">
  <div itemscope itemprop="includedInDataCatalog" itemtype="http://schema.org/DataCatalog">
    <meta itemprop="name" content="TensorFlow Datasets" />
  </div>
  <meta itemprop="name" content="cifar100" />
  <meta itemprop="description" content="This dataset is just like the CIFAR-10, except it has 100 classes containing 600 images each. There are 500 training images and 100 testing images per class. The 100 classes in the CIFAR-100 are grouped into 20 superclasses. Each image comes with a &quot;fine&quot; label (the class to which it belongs) and a &quot;coarse&quot; label (the superclass to which it belongs)." />
  <meta itemprop="url" content="https://www.tensorflow.org/datasets/catalog/cifar100" />
  <meta itemprop="sameAs" content="https://www.cs.toronto.edu/~kriz/cifar.html" />
</div>

# `cifar100`

This dataset is just like the CIFAR-10, except it has 100 classes containing 600
images each. There are 500 training images and 100 testing images per class. The
100 classes in the CIFAR-100 are grouped into 20 superclasses. Each image comes
with a "fine" label (the class to which it belongs) and a "coarse" label (the
superclass to which it belongs).

*   URL:
    [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
*   `DatasetBuilder`:
    [`tfds.image.cifar.Cifar100`](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/image/cifar.py)
*   Version: `v1.3.1`
*   Size: `160.71 MiB`

## Features
```python
FeaturesDict({
    'coarse_label': ClassLabel(shape=(), dtype=tf.int64, num_classes=20),
    'image': Image(shape=(32, 32, 3), dtype=tf.uint8),
    'label': ClassLabel(shape=(), dtype=tf.int64, num_classes=100),
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
