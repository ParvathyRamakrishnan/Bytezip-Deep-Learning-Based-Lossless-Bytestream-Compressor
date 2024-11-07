# ByteZip: Deep Learning-Based Lossless Bytestream Compressor

### Parvathy Ramakrishnan P, Satyajit Das  
#### Indian Institute of Technology Palakkad

---

This repository contains code and resources related to **ByteZip**, a deep learning-based approach for efficient lossless compression of structured byte streams. ByteZip leverages advanced machine learning model, including autoencoders and mixture density networks, to compress bytestreams losslessly. In contrast to other deep learning based lossless compressors, our approach follows a hierarchical probablistic approach and reduces the overhead of sequential processing. This work was presented at the **2024 International Joint Conference on Neural Networks (IJCNN)**.

## Citation

If you use ByteZip in your research or development, please cite our paper:

> **P. R. P and S. Das**, "ByteZip: Efficient Lossless Compression for Structured Byte Streams Using DNNs," *2024 International Joint Conference on Neural Networks (IJCNN)*, Yokohama, Japan, 2024, pp. 1-8, doi: [10.1109/IJCNN60899.2024.10650523](https://doi.org/10.1109/IJCNN60899.2024.10650523).

## Requirements
Python 3.6+
Libraries: PyTorch 1.1, numpy, torchac, scikit-learn

## Instructions
Sorce code is available in folder src. 

Copy the train and test data into data folder.
#### Neural Network Model
Define model and network parameters according to data in script multiscalemodel.py
#### Training: 
Copy train dataset in data folder and train model using script Training_mutiscalemodel.py

Save the trained model
#### Evaluation
To test lossless compression and decompression using trained model use script Compress_multiscalemodel.py



