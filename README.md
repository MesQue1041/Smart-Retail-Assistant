# Smart Retail Checkout Assistant

A computer vision and AI-based retail checkout system that detects multiple fruit and vegetable items from images, retrieves product metadata from an ontology, and generates a readable receipt using a local large language model.

## Overview

Traditional barcode-based checkout systems require each item to be scanned manually, which slows down the billing process. This project explores a smarter checkout workflow by using object detection to identify products directly from images.

The system uses two deep learning models, YOLOv8s and SSD300, to detect retail items from images. It also includes RDF/SPARQL-based ontology lookup and a Gemma model running locally through Ollama to generate receipts and semantic verification outputs.

## Features

- Multi-item object detection from retail images
- Two detection models: YOLOv8s and SSD300
- Merging of packaged and unpackaged product variants into unified classes
- Ontology-based product metadata lookup
- Receipt generation using a local LLM through Ollama
- Visual outputs including class distribution charts, confusion matrices, training curves, and comparison graphs

## Dataset

The project uses the Fruits and Vegetable Detection for YOLOv4 dataset from Kaggle. The original dataset contains 4,592 labelled images across 14 classes, which were merged into 8 final product classes for the retail checkout scenario.

### Final Classes

- Banana
- Blackberries
- Raspberry
- Lemon
- Grapes
- Tomato
- Apple
- Chilli

### Dataset Split

- Train: 3,214 images
- Validation: 688 images
- Test: 690 images

## Models

### YOLOv8s

YOLOv8s was trained using the Ultralytics framework with 50 epochs, batch size 8, and image size 640x640.

#### Test Results

- Precision: 0.9691
- Recall: 0.9703
- F1-score: 0.9697
- mAP@0.5: 0.9904
- mAP@0.5:0.95: 0.8334

### SSD300

SSD300 was trained using a custom PyTorch pipeline with 15 epochs, batch size 8, and image size 300x300.

#### Test Results

- Precision: 0.9413
- Recall: 0.9990
- F1-score: 0.9689
- mAP@0.5: 0.9413
- mAP@0.5:0.95: 0.6457

## Technology Stack

- Python
- PyTorch
- Ultralytics YOLOv8
- SSD300
- OpenCV
- Matplotlib
- Pandas
- Flask
- RDF / SPARQL
- rdflib
- Ollama
- Gemma 3


