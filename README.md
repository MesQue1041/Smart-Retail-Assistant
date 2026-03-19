# Smart Retail Checkout Assistant

A computer vision and AI-based retail checkout system that detects multiple fruit and vegetable items from images, retrieves product metadata from an ontology, and generates a readable receipt using a local large language model. [file:14][file:65]

## Overview

Traditional barcode-based checkout systems require each item to be scanned manually, which slows down the billing process. This project explores a smarter checkout workflow by using object detection to identify products directly from images. [file:14]

The system uses two deep learning models, YOLOv8s and SSD300, to detect retail items from images. It also includes RDF/SPARQL-based ontology lookup and a Gemma model running locally through Ollama to generate receipts and semantic verification outputs. [file:14][file:65]

## Features

- Multi-item object detection from retail images. [file:14][file:65]
- Two detection models: YOLOv8s and SSD300. [file:14][file:65]
- Merging of packaged and unpackaged product variants into unified classes. [file:14][file:65]
- Ontology-based product metadata lookup. [file:14][file:65]
- Receipt generation using a local LLM through Ollama. [file:65]
- Visual outputs including class distribution charts, confusion matrices, training curves, and comparison graphs. [file:65]

## Dataset

The project uses the Fruits and Vegetable Detection for YOLOv4 dataset from Kaggle. The original dataset contains 4,592 labelled images across 14 classes, which were merged into 8 final product classes for the retail checkout scenario. [file:14][file:65]

### Final Classes

- Banana
- Blackberries
- Raspberry
- Lemon
- Grapes
- Tomato
- Apple
- Chilli [file:14][file:65]

### Dataset Split

- Train: 3,214 images [file:65]
- Validation: 688 images [file:65]
- Test: 690 images [file:65]

## Models

### YOLOv8s

YOLOv8s was trained using the Ultralytics framework with 50 epochs, batch size 8, and image size 640x640. [file:14][file:65]

#### Test Results

- Precision: 0.9691 [file:65]
- Recall: 0.9703 [file:65]
- F1-score: 0.9697 [file:65]
- mAP@0.5: 0.9904 [file:65]
- mAP@0.5:0.95: 0.8334 [file:65]

### SSD300

SSD300 was trained using a custom PyTorch pipeline with 15 epochs, batch size 8, and image size 300x300. [file:14][file:65]

#### Test Results

- Precision: 0.9413 [file:65]
- Recall: 0.9990 [file:65]
- F1-score: 0.9689 [file:65]
- mAP@0.5: 0.9413 [file:65]
- mAP@0.5:0.95: 0.6457 [file:65]

## Technology Stack

- Python [file:14]
- PyTorch [file:14]
- Ultralytics YOLOv8 [file:14][file:65]
- SSD300 [file:14][file:65]
- OpenCV [file:14]
- Matplotlib [file:65]
- Pandas [file:65]
- Flask [file:14]
- RDF / SPARQL [file:14]
- rdflib [file:65]
- Ollama [file:14][file:65]
- Gemma 3 [file:14][file:65]

