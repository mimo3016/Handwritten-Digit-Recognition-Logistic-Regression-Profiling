# Handwritten-Digit-Recognition-Logistic-Regression-Profiling
This project explores the mathematical boundaries of linear classifiers on non-linear, high-dimensional human handwriting. Instead of jumping straight to a complex neural network, this system was designed to benchmark a classical linear baseline, trace how it draws decision hyperplanes, and deeply analyse boundary region failures.

_Key Features:_
**Custom CV Preprocessing Pipeline:** Programmatically crops, deskewes, and extracts 1,200 individual digit cells from physical paper scans.
**Multinomial Logistic Regression**: Implements a multiclass Softmax model using the quasi-Newton lbfgs optimisation solver.
**Mathematical Boundary Profiling:** Interrogates the confidence distributions of Softmax probabilities to map out high-uncertainty decision regions.
**Detailed Error Analysis:** Investigates how purely linear models fail to interpret topological features (e.g., classifying a "0" with a gap at the top as a "3").

_Project Structure:_

├── dataset/                 # 1,200 cropped 28x28 digit images & labels.csv
├── raw_sheets/              # Original high-resolution scanner/mobile inputs
├── process_sheets.py        # OpenCV pipeline for cell isolation & grid cropping
├── Notebook.ipynb           # Model training, evaluation, & probability profiling
└── README.md                # Project documentation

1. **The Preprocessing Pipeline (CV)**                                                                                                                              The dataset consists of 1,200 custom handwritten samples spanning digits 0 to 5 (exactly 200 balanced samples per class). To simulate real-world challenges, handwriting parameters were dynamically varied with shifting line thicknesses (pen pressure), digit scales, and rotational tilts.                        Extraction Strategy:                                                                                                                                        Physical handwriting was captured on custom A4 grid sheets and digitised using a document scanner.                                                          process_sheets.py uses OpenCV to identify the grid corners, correct perspective distortion, and segment the sheet into clean, individual cell images.      Outliers and crop failures caused by scan skewing were isolated via an automated debug pipeline to guarantee high-integrity input arrays.

2.  **Model & Optimisation Architecture:**                                                                                                              Model Multinomial Logistic Regression (Softmax Activation)                                                                                                    Optimisation Solver: lbfgs (Limited-memory Broyden–Fletcher–Goldfarb–Shanno) optimizing cross-entropy loss.                                        Hyperparameters: max_iter=5000 to ensure complete convergence despite the structural noise inherent in handwriting.                                      Validation Split: Stratified 80/20 train/test split (960 training samples, 240 test samples with exactly 40 balanced test samples per digit).

3.  **Performance & Empirical Insights**                                                                                                                           Key Metrics                                                                                                                                                 Overall Test Accuracy: 52.5%                                                                                                                                 Highest Performing Class (Precision): Digit 0 (Precision: 0.83, Recall: 0.75), owing to its distinct circular topology in the 784-dimensional feature space.  Lowest Performing Class (Recall): Digit 5 (Recall: 0.35), heavily confused with 3 and 1 due to highly similar horizontal and vertical stroke alignments.        The Most Confused Pair: 1 vs 2                                                                                                                                  The model incorrectly flipped 1 as a 2 nine times, and a 2 as a 1 ten times. Because linear classifiers partition space using straight hyperplanes, they struggle to separate the straight vertical backbones common to both handwritten styles.

4.  **Deep-Dive: Probability Boundary Analysis**                                                                                                                    One of the core objectives of this project was to analyse how the Softmax confidence distribution behaves during a classification.                          Correct Classification (High Certainty)                                                                                                                          For a cleanly drawn 3, the model generated a highly "peaked" distribution:                                                                                    P(3): 89.9% (True Class)                                                                                                                                      P(Next Best Guess): 8.8%                                                                                                                                  Takeaway: The feature vector lay far from any decision hyperplanes, allowing the model to make a mathematically certain prediction.

Misclassification (High Ambiguity)
When analysing a true 1 misclassified as a 2, the probabilities were highly distributed:
P(2): 48.6% (Predicted Class)
P(1): 42.6% (True Class)
Takeaway: The marginal $6\%$ difference proves that the sample was situated deep within a decision boundary region. The introduction of handwriting tilts and tails forced the linear model to essentially "flip a coin."

The Topological Blindspot (The 0 vs 3 Case)
A handwritten 0 with a slight gap at its top loop was misclassified as a 3 with $57\%$ confidence.
Because linear Softmax regression relies on pixel intensities at fixed grid locations rather than spatial relationships, it cannot natively comprehend "closed-loop topology". It mathematically interpreted the open arcs of the incomplete 0 as the open curves of a 3.

How to Setup & Run

Prerequisites
Make sure you have Python 3.8+ and the necessary libraries installed:

## 📦 Dataset Availability

Due to GitHub's file size limitations, the raw image datasets are not hosted directly in this repository. 

* **Download the Dataset:** [Click here to download the processed 1,200 digit images (ZIP)](#) <!-- https://drive.google.com/drive/folders/1vFswIG4-rnj1fy-mjonsvMMkOLIrZB3A?usp=drive_link -->
* **Download Raw Sheets:** [Click here to download the 6 original A4 scanned grid sheets (ZIP)](#) <!-- https://drive.google.com/drive/folders/1vFswIG4-rnj1fy-mjonsvMMkOLIrZB3A?usp=drive_link -->

Once downloaded, extract the folders into the root directory of this project to run the pipeline:
```text
├── dataset/         <-- Extract processed images here
├── raw_sheets/      <-- Extract raw scans here

Bash
pip install numpy pandas scikit-learn opencv-python matplotlib jupyter

Running the Pipeline(Optional) 
Re-process Raw Scans: To run the computer vision extraction script over the raw grids:
Bash
python process_sheets.py --input raw_sheets --out dataset --debug

Train and Profile the Model:
Launch the Jupyter notebook to view the step-by-step training process, confusion matrix generation, and probability visualisation
Bash
jupyter notebook Notebook.ipynb
