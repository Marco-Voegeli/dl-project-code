# normal-difference-prediction
Predict normal difference for each pixel point.

## Usage
First, download and process the data as described in `../data`, and link it to here by `ln -s <PATH-TO-DATASET> data`. Then, go through the `regression.ipynb`.

## Visualization
We provide not only normal difference comparison but also per-line classification (on whether structural) result comparison between ground truth ([LSD(Pytlsd)](https://github.com/iago-suarez/pytlsd)) vs predicted for each image of the dataset. Go through the `visualization.ipynb`.

## Reference
[Training Pipeline](https://github.com/qubvel/segmentation_models.pytorch "Pytorch Semantic Segmentation Library"), and [Configuration](https://github.com/CSAILVision/semantic-segmentation-pytorch "MIT Semantic Segmentation")