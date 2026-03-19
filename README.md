Smart Retail Checkout Assistant
A computer vision and AI-based retail checkout system that detects multiple fruit and vegetable items from images, retrieves product metadata from an ontology, and generates a readable receipt using a local large language model.

Project Overview
Traditional barcode-based checkout systems are reliable, but they require each item to be scanned manually. This project explores a smarter checkout workflow by using object detection to identify products directly from images.

The system was built using two deep learning models, YOLOv8s and SSD300, and compares their performance on a retail fruit and vegetable dataset. The final pipeline also includes RDF/SPARQL-based ontology lookup and a Gemma model running locally through Ollama.

Features
Detects multiple retail items from a single image.

Uses two object detection models: YOLOv8s and SSD300.

Merges packaged and unpackaged variants into unified product classes.

Retrieves product details such as price, unit, category, and stock from an ontology.

Generates natural language receipts using a local LLM with Ollama.

Produces evaluation outputs including per-class metrics, confusion matrices, training curves, and model comparison charts.

Dataset
The project uses the Fruits and Vegetable Detection for YOLOv4 dataset from Kaggle. The original dataset contains 4,592 labelled images across 14 classes, and these were merged into 8 final product classes for the checkout use case.

Final classes used in this project:

Banana

Blackberries

Raspberry

Lemon

Grapes

Tomato

Apple

Chilli

Dataset split:

Train: 3,214 images

Validation: 688 images

Test: 690 images
​

Models Used
YOLOv8s
YOLOv8s was trained using the Ultralytics framework with 50 epochs, 640x640 input size, batch size 8, and AdamW optimizer.

Test set results:

Precision: 0.9691

Recall: 0.9703

F1-score: 0.9697

mAP@0.5: 0.9904

mAP@0.5:0.95: 0.8334
​

SSD300
SSD300 was trained using a custom PyTorch pipeline with 15 epochs, 300x300 input size, batch size 8, and SGD optimizer.

Test set results:

Precision: 0.9413

Recall: 0.9990

F1-score: 0.9689

mAP@0.5: 0.9413

mAP@0.5:0.95: 0.6457
​

Technology Stack
Python

PyTorch

Ultralytics YOLOv8

SSD300

OpenCV

Matplotlib

Pandas

Flask

RDF / SPARQL

rdflib

Ollama

Gemma 3

Project Structure
bash
retail_cv_project/
├── data/
│   ├── processed/
│   │   ├── train/
│   │   ├── valid/
│   │   └── test/
├── models/
│   ├── yolov8s_retail/
│   └── ssd_best.pth
├── results/
│   ├── class_distribution.png
│   ├── yolov8_per_class_metrics.png
│   ├── ssd_per_class_metrics.png
│   ├── modelcomparison.png
│   ├── confusion_matrices/
│   ├── sample_predictions/
│   └── receipts/
├── retail_ontology.ttl
└── notebook_or_scripts
Workflow
Prepare the raw dataset and collect image-label pairs.

Merge the original 14 classes into 8 final retail product classes.

Split the data into training, validation, and test sets.

Train YOLOv8s and SSD300 separately.

Evaluate both models using precision, recall, F1-score, and mAP.

Build an ontology to store product metadata.

Use SPARQL queries to retrieve product information after detection.

Generate a receipt and semantic verification output using Gemma through Ollama.

Installation
Clone the repository:

bash
git clone https://github.com/your-username/smart-retail-checkout-assistant.git
cd smart-retail-checkout-assistant
Install the required Python packages:

bash
pip install ultralytics opencv-python matplotlib seaborn scikit-learn tqdm albumentations
pip install torch torchvision torchmetrics rdflib ollama
Running the Project
1. Prepare the dataset
Update the dataset paths in the code so they point to your local copy of the Kaggle dataset.

2. Train YOLOv8s
Run the YOLO training section in the notebook or script to generate the trained weights and evaluation outputs.

3. Train SSD300
Run the SSD300 training section to train the second detector and save the best model checkpoint.

4. Build ontology
Run the ontology section to create the Turtle file and populate product metadata.

5. Run the full pipeline
Run the final pipeline code to:

detect items from an image

query ontology data

generate a receipt

generate semantic verification text
​

Sample Outputs
This project produces:

Class distribution charts

Bounding box distribution charts

Training curves for YOLOv8s and SSD300

Per-class metric charts

Confusion matrices

Sample predictions on test images

YOLOv8s vs SSD300 comparison charts

Ontology query outputs

LLM-generated receipts and semantic verification text
​

Key Findings
YOLOv8s achieved the best overall detection performance, especially in precision and localisation quality, with a test mAP@0.5 of 0.9904 and mAP@0.5:0.95 of 0.8334. SSD300 achieved much higher recall at 0.9990, meaning it missed very few objects, but its localisation accuracy was lower than YOLOv8s.
​

This makes YOLOv8s the better final model for the retail checkout scenario, while SSD300 remains a strong baseline due to its very high recall.
​

Future Improvements
Expand the dataset to include more supermarket products

Improve small object and overlapping object detection

Deploy the system on embedded hardware for real-time checkout

Add payment integration and more advanced product reasoning

Extend the ontology with live inventory and pricing updates
