# Traffic Sign Detection
This is the code for the paper

**[Evaluation of deep neural networks for traffic sign detection systems](https://doi.org/10.1016/j.neucom.2018.08.009)**
<br>
[Álvaro Arcos-García](https://scholar.google.com/citations?user=gjecl3cAAAAJ),
[Juan Antonio Álvarez-García](https://scholar.google.com/citations?user=Qk79xk8AAAAJ),
[Luis M. Soria-Morillo](https://scholar.google.com/citations?user=poBDpFkAAAAJ)
<br>

The paper addresses the problem of traffic sign detection analysing the state-of-the-art of several object-detection systems (Faster R-CNN, R-FCN, SSD, and YOLO V2) combined with various feature extractors (Resnet V1 50, Resnet V1 101, Inception V2, Inception Resnet V2, Mobilenet V1, and Darknet-19). We aim to explore the properties of these object-detection models which are modified and specifically adapted to the traffic sign detection problem domain by means of transfer learning. In particular, various publicly available object-detection models that were pre-trained on the Microsoft COCO dataset are fine-tuned on the [German Traffic Sign Detection Benchmark](http://benchmark.ini.rub.de/?section=gtsdb) dataset. The evaluation and comparison of these models include key metrics, such as the mean average precision (mAP), memory allocation, running time, number of floating point operations, number of parameters of the model, and the effect of traffic sign image sizes.
![Detection example](detection_example.png?raw=True "Detection example")

We provide:
- Several [pretrained models](#pretrained-models).
- [Experiment results](#results).
- Test code to [run the model on new images](#running-on-new-images).
- Instructions for [training your model](#training).
- Instructions for [evaluating your model](#evaluation).
- Scripts to create GTSDB TFRecords.

If you find this code useful in your research, please cite:

```
"Evaluation of deep neural networks for traffic sign detection systems."
Álvaro Arcos-García, Juan A. Álvarez-García, Luis M. Soria-Morillo. Neurocomputing 316 (2018) 332-344.
```
\[[link](https://doi.org/10.1016/j.neucom.2018.08.009)\]\[[bibtex](
https://scholar.googleusercontent.com/citations?view_op=export_citations&user=gjecl3cAAAAJ&citsig=AMstHGQAAAAAW88AbhXW-l4i_qhFLdvy09TVYklB-yqs&hl=en)\]

## Requirements
This project is implemented in [Tensorflow](https://www.tensorflow.org/) and it is based on two repositories: [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) and [darkflow](https://github.com/thtrieu/darkflow).

## Pretrained models
You can download TFRecords of the GTSDB dataset from [Google Drive](https://drive.google.com/open?id=1hKdjTsiFm_vdtZPdto0QJihThGFOPEkq). Unzip it to the `gtsdb_data` folder.
You can download pre-trained models from Google Drive. Unzip them to the `models` folder.
- [faster_rcnn_inception_resnet_v2_atrous](https://drive.google.com/open?id=12vLvA9wyJ9lRuDl9H9Tls0z5jsX0I0Da)
- [faster_rcnn_inception_v2](https://drive.google.com/open?id=1LRCSWIkX_i6ijScMfaxSte_5a_x9tjWF)
- [faster_rcnn_resnet_101](https://drive.google.com/open?id=15OxyPlqyOOlUdsbUmdrexKLpHy1l5tP9)
- [faster_rcnn_resnet50](https://drive.google.com/open?id=1aEqlozB_CzhyJX_PO6SSiM-Yiv3fuO8V)
- [rfcn_resnet101](https://drive.google.com/open?id=1eWCDZ5BxcEa7n_jZmWUr2kwHPBi5-SMG)
- [ssd_inception_v2](https://drive.google.com/open?id=1TKMd-wIZJ1aUcOhWburm2b6WgYnP0ZK6)
- [ssd_mobilenet_v1](https://drive.google.com/open?id=1U31RhUvE1Urr5Q92AJynMvl-oFBVRxxg)
- [yolo_v2](https://drive.google.com/open?id=1wqWgHqcwtjXTXvKZgkzfbTvdpDAy8G85) (Updated 19/02/2019: Add missing file 'yolo_v2-50500.data-00000-of-00001')

## Results
### Overview
Our findings show that Faster R-CNN Inception Resnet V2 obtains the best mAP, while R-FCN Resnet 101 strikes the best trade-off between accuracy and execution time. YOLO V2 and SSD Mobilenet merit a special mention, in that the former achieves competitive accuracy results and is the second fastest detector, while the latter, is the fastest and the lightest model in terms of memory consumption, making it an optimal choice for deployment in mobile and embedded devices.

| model                            | parameters | flops         | memory_mb    | total_exec_millis | accelerator_exec_millis | cpu_exec_millis | mAP   |
|----------------------------------|------------|---------------|--------------|-------------------|-------------------------|-----------------|-------|
| Faster R-CNN Resnet 50           | 43337242   | 533575386662  | 5256.454615  | 104.0363553       | 75.93395395             | 28.10240132     | 91.52 |
| Faster R-CNN Resnet 101          | 62381593   | 625779295782  | 6134.705805  | 123.2729175       | 90.33714433             | 32.9357732      | 95.08 |
| Faster R-CNN Inception V2        | 12891249   | 120621363525  | 2175.206857  | 58.53338971       | 38.76813971             | 19.76525        | 90.62 |
| Faster R-CNN Inception Resnet V2 | 59412281   | 1837544257834 | 18250.446008 | 442.2206796       | 366.1586796             | 76062           | 95.77 |
| R-FCN Resnet 101                 | 64594585   | 269898731281  | 3509.75153   | 85.45207971       | 52.40321739             | 33.04886232     | 95.15 |
| SSD Mobilenet                    | 5572809    | 2300721483    | 94.696119    | 15.14525          | 4.021267857             | 11.12398214     | 61.64 |
| SSD Inception V2                 | 13474849   | 7594247747    | 284.512918   | 23.74428378       | 9.393405405             | 14.35087838     | 66.10 |
| YOLO V2                          | 50588958   | 62780021160   | 1318.108256  | 21.4810122        | 18.13923171             | 3.341780488     | 78.83 |

![Results plot](results_radar_plot.png?raw=True "Results plot")

### Image size comparison

| model                            | small | medium | large | overall |
|----------------------------------|-------|--------|-------|---------|
| Faster R-CNN Resnet 50           | 53.57 | 86.95  | 86.72 | 91.52   |
| Faster R-CNN Resnet 101          | 70.89 | 94.17  | 88.87 | 95.08   |
| Faster R-CNN Inception V2        | 56.72 | 81.02  | 88.53 | 90.62   |
| Faster R-CNN Inception Resnet V2 | 68.60 | 86.62  | 82.10 | 95.77   |
| R-FCN Resnet 101                 | 60.37 | 82.03  | 79.56 | 95.15   |
| SSD Mobilenet                    | 22.13 | 55.32  | 82.06 | 61.64   |
| SSD Inception V2                 | 26.85 | 64.71  | 78.76 | 66.1    |
| YOLO V2                          | 42.93 | 78.99  | 75.67 | 78.83   |

![mAP vs image size](mAP_vs_image_size.png?raw=True "mAP vs image size")

## Running on new images
We provide a [Jupyter Notebook](Run_models_on_new_images.ipynb) with instructions to run our pre-trained models on new images.

## Training
Config files needed to train our models are included within each zip file that you can download from the [pre-trained models](#pretrained-models) section. Minimal changes were done to the base source code of tensorflow and darkflow so you should be able to reproduce our results using the last versions of the aforementioned repositories.
For instance, to train a SDD Mobilenet model you can run the following command lines:
```bash
cd ~/tensorflow/models/research
python3 object_detection/train.py --logtostderr --pipeline_config_path models/ssd_mobilenet_v1/ssd_mobilenet_v1_gtsdb3.config --train_dir models/ssd_mobilenet_v1/train/
```

## Evaluation
We provide a [Jupyter Notebook](Evaluation.ipynb) with instructions to evaluate the results obtained by our pre-trained models.


## Acknowledgements
The source code of this project is mainly based on [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) and [darkflow](https://github.com/thtrieu/darkflow).
