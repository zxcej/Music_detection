## Music note classifier

This is a simple music symbol classifier. It uses a Vision Transformer (ViT) to classify music notes. The model is trained on the HOMUS dataset.


### Usage

The model to use is 'output_2'. 

```python

from transformers import pipeline

classifier = pipeline('image-classification', model='output_2')
image = 'path/to/image'
classifier(image)

```



Please refer to test.ipynb for an example of how to use the model.





