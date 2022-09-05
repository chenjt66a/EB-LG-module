# EB-LG-module
This repository includes the implementation of EB-LG module.

# Install
The code has been tested on one configuration:

        pytorch 1.0.0, cuda 10.0
  
Please refer to AdaptConv / DGCNN to install other required packages.

# Data
Please download the ModelNet40 and ScanObjectNN datasets and place them to:

        cls/data/datasets.

# Usage
To train or test a model by:

        python train.py
        
# Acknowledgement
The code is built on PointNet++ / DGCNN / AdaptConv. We thank the authors for sharing the codes.
