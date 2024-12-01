# **AIRBNB IMAGE FORECASTER**

## This project aims to create a Deep Learning model from 0 and using neural dense networks combining tabular and image data to predict real estate prices🏫

Project and Arquitecture structure:

1. 💻Data Preprocessing:

Downloaded and cleaned datasets, removing rows with missing images to maintain consistency between image and tabular data.
Applied normalization and resizing to both tabular and image datasets, ensuring consistent preprocessing pipelines.
Ensured that all preprocessing and transformations applied to the training data were reused for validation and test datasets to prevent data leakage.

2. ✅Modeling:

Designed and implemented a custom dense neural network for tabular data using features like Property Type, Room Type, and Bathrooms.
Developed and fine-tuned a model using pretrained architectures for image data, leveraging transfer learning and justifying the choice based on pretraining relevance.
Selected appropriate loss functions (e.g., regression or classification-specific) and optimized hyperparameters to improve model performance while preventing overfitting.

3. ➕Model Fusion:

Experimented with early fusion techniques by combining features extracted from images with tabular data into a single input vector for predictions.
Explored late fusion approaches by generating separate predictions from tabular and image models, then integrating them using ensemble techniques (e.g., Random Forests or Neural Networks).

4. 👍Iterative Baseline Improvement:

Established a baseline model for comparison, iteratively adding enhancements and evaluating their impact on the validation set.

5. 🪄Post-Processing and Predictions:

Applied post-processing steps to predictions, such as reverting normalized outputs to their original range for real-world interpretability.
Evaluated and refined predictions to ensure they met the objectives of regression or classification tasks.

7. 📈Advanced Techniques:

Tackled missing values and engineered relevant features from the dataset to enhance model inputs.
Used scalable techniques like MinMaxScaler for regression and classification labeling strategies, optimizing input for the chosen model type.

_____________________________________

✔️File Insights

1. 1_Imagenes_descarga_preprocss.ipynb
Downloaded, resized, and normalized image data (224x224), applying min-max scaling and channel-wise normalization (mean and standard deviation) for optimized input to the model.
Calculated and stored mean and standard deviation per channel (R, G, B) from training images to ensure consistent normalization across training, validation, and test sets.
Synchronized preprocessed image datasets with tabular data, addressing record mismatches and saving updated datasets.
Implemented deep and efficient data handling workflows, ensuring compatibility and reducing preprocessing overhead for multimodal deep learning tasks.

2. 2_Modelado_tabulares
Designed a tailored neural network with 64 and 32 neurons in hidden layers and Dropout regularization to predict price_log.
Implemented a Learning Rate Finder to optimize training performance efficiently.
Applied imputation, normalization, and feature engineering to prepare the dataset for modeling, ensuring data integrity.
Built an end-to-end regression pipeline in TensorFlow and Keras, showcasing expertise in model architecture, training, and evaluation.

3. 2_Modelado_imagenes_full
Adapted a pre-trained ResNet50 model to predict real estate prices from interior images.
Added dense layers (512, 256 neurons), dropout for regularization, and a linear output layer for regression.
Normalized and preprocessed images using ResNet50-specific techniques, ensuring reproducibility.
Used Adam optimizer, MSE loss, and MAE metric for accurate and interpretable predictions.

## 4. **3.Early_Fusion_imagenes-tabular (still working on)**⬅️
Trained and evaluated multiple pre-trained models, including ResNet50, EfficientNet, and MobileNet, showcasing expertise in leveraging various architectures.
Combined image features with tabular data through a Concatenate Layer, enabling multi-modal learning.
Extended pre-trained models with custom dense layers to adapt them for regression tasks, ensuring compatibility with fused features.
Standardized tabular data to match image feature scales, ensuring seamless integration across modalities.
Used EfficientNet to extract image features and integrated cluster-based outlier detection for price predictions, improving robustness against anomalous data.
Achieved promising results with validation metrics indicating potential for further optimization, reflecting adaptability across models and data types.

5. PREPROCESS_fd_val_.ipynb
Aligned validation dataset preprocessing with training and test sets, ensuring feature consistency across all splits.

6. PREPROCESS_fd_test.ipynb
Finalized preprocessing of test data, ensuring consistency with validation and training datasets for seamless model integration.


