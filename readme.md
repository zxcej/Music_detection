## Music note classifier

This is a simple music symbol classifier. It uses a Vision Transformer (ViT) to classify music notes. The model is trained on the [HOMUS dataset](https://github.com/apacha/Homus).

##### Evaluation Metrics

| Metric                   | Value     |
|--------------------------|-----------|
| epoch                    | 3.0       |
| eval_accuracy            | 0.8829    |
| eval_loss                | 0.4572    |
| eval_runtime             | 0:00:13.98|
| eval_samples_per_second  | 162.981   |
| eval_steps_per_second    | 20.373    |



### Dependencies 

* Python 3.9 or higher
* PyTorch

The model was trained using the Hugging Face Transformers library. You can install the library using the following command:

```bash
pip install transformers evaluate datasets
```

### Usage

The model to use is 'output_2'. 

```python

from transformers import pipeline

classifier = pipeline('image-classification', model='output_2')
image = 'path/to/image'
classifier(image)

```
A sample output for an F-clef image:
![](C:\Users\x_zhu202\PycharmProjects\Music_detection\Unseen_test\f_clef.png)


```json
[
  {"score": 0.8796402812004089, "label": "F-Clef"},
  {"score": 0.018305785953998566, "label": "Common-Time"},
  {"score": 0.0076107243075966835, "label": "Sixty-Four-Rest"},
  {"score": 0.006926759146153927, "label": "Thirty-Two-Rest"},
  {"score": 0.0068232957273721695, "label": "Eighth-Rest"}
]
```



Please refer to test.ipynb for an example of how to use the model.

### Training

To train the Vision Transformer (ViT) model on the HOMUS dataset, you can use the run_image_classification.py script provided by the Hugging Face Transformers library. The following command demonstrates how to train and evaluate the model using the script:

```bash
python run_image_classification.py \
  --train_dir ./homus_data \
  --output_dir ./outputs_2/ \
  --remove_unused_columns False \
  --do_train True \
  --do_eval True
```

* `--train_dir`: The directory containing your training dataset. In this case, the dataset is stored in the homus_data folder.
* `--output_dir`: The directory where the script will save the trained model, evaluation results, and other outputs. In this example, the output will be stored in the outputs_2 folder.
* `--remove_unused_columns`: This option determines whether to remove unused columns from the dataset. Set it to False to keep all columns in the dataset.
* `--do_train`: Set this option to True to enable model training. The script will train the model on the specified dataset.
* `--do_eval`: Set this option to True to enable model evaluation. After training the model, the script will evaluate its performance on the test set.


For more parameters and options, visit the [Hugging Face Transformers Image Classification example](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) on GitHub.

#### Limitation

Please note that the current model has a limitation: it requires musical symbols to be on a **white background** for accurate classification.



