# Table parsing
This code is heavily influenced from <b>https://medium.com/@djajafer/pdf-table-extraction-with-keras-retinanet-173a13371e89</b>. The major difference is that the mentioned link uses keras based RetinaNet object detection but this repository uses pytorch based retinanet implementation (<b>https://github.com/yhenon/pytorch-retinanet</b>)

The problem statement of detecting tables in pdf files is handled as an Object detection problem. One such CNN based model is <b>RetinaNet</b>. To understand how RetinaNet works, please go through the following blogs:
* https://blog.zenggyu.com/en/post/2018-12-05/retinanet-explained-and-demystified/
* https://developers.arcgis.com/python/guide/how-retinanet-works/

## Evaluation of model
To understand how to evaluate object detection models, please read the following link:
https://blog.zenggyu.com/en/post/2018-12-16/an-introduction-to-evaluation-metrics-for-object-detection/. 
Based upon it, we use <b>mean Average Precision(mAP)</b> to evaluate the fine tuned model.

## Creating data for training the model:
* <b>ConvertPDF2Img.ipynb</b> : Converts PDF to JPG images and saves on disk.
* <b>ConvertVOC2CSVFormat.ipynb</b>: Converts PASCALVOC annotation format(xml) to csv format as mentioned in https://github.com/yhenon/pytorch-retinanet
* <b>ParseTable.ipynb</b>: Notebook to train and infer the model.

## Tool for annotating the data:
* https://www.makesense.ai/: Provides PASCALVOC annnotation format

## Command for fine tuning the model:
python pytorch-retinanet/train.py --dataset csv --csv_train Images/retinanet_train.csv --csv_val Images/retinanet_val.csv --csv_classes Images/retinanet_classes.csv --depth 50 --epochs=20

## Link to fine-tuned model
https://drive.google.com/file/d/10H5etyY8x3fEfR6vAs3YNct5RGSWB0XI/view?usp=sharing

## Command for inferencing:
python pytorch-retinanet/visualize.py --dataset csv --csv_classes Images/retinanet_classes.csv --csv_val Images/retinanet_test_new.csv --model csv_retinanet_15.pt --output_path Images/test_new_tagged/

## Model performance
Our fine-tuned model achieves <b>mAP=0.7</b>

## Examples
![img1](https://github.com/hardiksahi/DeepLearning/blob/master/CV/Table_Parsing/examples/10.jpg)
![img2](https://github.com/hardiksahi/DeepLearning/blob/master/CV/Table_Parsing/examples/13.jpg)
![img4](https://github.com/hardiksahi/DeepLearning/blob/master/CV/Table_Parsing/examples/14.jpg)
![img6](https://github.com/hardiksahi/DeepLearning/blob/master/CV/Table_Parsing/examples/15.jpg)
![img7](https://github.com/hardiksahi/DeepLearning/blob/master/CV/Table_Parsing/examples/5.jpg)
