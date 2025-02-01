# **AIRBNB IMAGE FORECASTER**

## This project aims to create a Deep Learning model from 0 and using neural dense networks and pre-trained models combining tabular and image data to predict real estate prices🏫

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

📜 **1. 1_Imagenes_descarga_preprocss.ipynb**

🔸Downloaded, resized, and normalized image data (224x224), applying min-max scaling and channel-wise normalization (mean and standard deviation) for optimized input to the model.

🔸Calculated and stored mean and standard deviation per channel (R, G, B) from training images to ensure consistent normalization across training, validation, and test sets.

🔸Synchronized preprocessed image datasets with tabular data, addressing record mismatches and saving updated datasets.

🔸Implemented deep and efficient data handling workflows, ensuring compatibility and reducing preprocessing overhead for multimodal deep learning tasks.


📜 **1_PREPROCESS_fd_val_.ipynb**

🔸Aligned validation dataset preprocessing with training and test sets, ensuring feature consistency across all splits.


📜 **1_PREPROCESS_fd_test.ipynb**

🔸Finalized preprocessing of test data, ensuring consistency with validation and training datasets for seamless model integration.


📜 **2. 1_Modelado_tabulares**
   
🔸Designed a tailored neural network with 64 and 32 neurons in hidden layers and Dropout regularization to predict price_log.

🔸Implemented a Learning Rate Finder to optimize training performance efficiently.

🔸Applied imputation, normalization, and feature engineering to prepare the dataset for modeling, ensuring data integrity.

🔸Built an end-to-end regression pipeline in TensorFlow and Keras, showcasing expertise in model architecture, training, and evaluation.

![test_modelostabulares](https://github.com/user-attachments/assets/d03f9242-5880-49c9-b2da-6a13e0bcd54d)


📜 **2. 2_Modelado_imagenes_full**
   
🔸Adapted a pre-trained ResNet50 model to predict real estate prices from interior images.

🔸Added dense layers (512, 256 neurons), dropout for regularization, and a linear output layer for regression.

🔸Normalized and preprocessed images using ResNet50-specific techniques, ensuring reproducibility.

🔸Used Adam optimizer, MSE loss, and MAE metric for accurate and interpretable predictions.

![modeladoimgs_Metrics](https://github.com/user-attachments/assets/f64b8553-8d99-41f2-9e0c-ee0951226846)


📜 **3. Early_Fusion_imagenes-tabular**

🔸Trained and evaluated multiple pre-trained models, including ResNet50, EfficientNet, and MobileNet, showcasing expertise in leveraging various architectures.

🔸Combined image features with tabular data through a Concatenate Layer, enabling multi-modal learning.

🔸Extended pre-trained models with custom dense layers to adapt them for regression tasks, ensuring compatibility with fused features.

🔸Standardized tabular data to match image feature scales, ensuring seamless integration across modalities.

🔸Used EfficientNet to extract image features and integrated cluster-based outlier detection for price predictions, improving robustness against anomalous data.

🔸Achieved promising results with validation metrics indicating potential for further optimization, reflecting adaptability across models and data types.


📜 **4. Late_Fusion_imgs_tab**
   
🔸Developed and optimized late fusion models combining traditional ML (Ridge, SVM, Random Forest) and deep learning (ResNet50, CNN with attention) to predict real-estate prices, improving model performance.

🔸Integrated tabular and image data using late fusion techniques, leveraging models like Random Forest, XGBoost, and CNNs to enhance prediction accuracy.

🔸Fine-tuned hybrid models, employing hyperparameter optimization and feature selection to maximize performance, including a hybrid CNN + attention mechanism model.

🔸Implemented robust training strategies with early stopping and model checkpoints to prevent overfitting, ensuring reproducibility and improved model generalization.

Evaluated model performance across multiple datasets (training, validation, test) with key metrics (MAE, RMSE, R²), delivering actionable insights into model effectiveness.
![image](https://github.com/user-attachments/assets/9b0f7fd9-782b-43a8-8f2a-f88018c4a0fe)


⏩⏪⏸️ **Project challenges and learnings:**

🔹Overfitting with early models: Overfitting occurred with poor validation/test performance. Solution: Applied Dropout, early stopping, and ReduceLROnPlateau to improve generalization.

🔹Log vs. Original Scale Performance: Good log scale results but weak original scale predictions. Solution: Used stacking with Random Forest and Neural Networks for better results across both scales.

🔹Hyperparameter tuning struggles: Difficulty finding optimal parameters led to suboptimal performance. Solution: Implemented Keras Tuner and GridSearch to refine hyperparameters and improve model results.

🔹Feature importance misunderstanding: Struggled with understanding which features had the most impact. Solution: Used SHAP to identify key features like accommodates and property_type_encoded to improve model focus.

🔹Handling Mixed Data Types: Challenge with combining numerical and categorical data effectively. Solution: Focused on normalization and consistent encoding, ensuring data was properly preprocessed for model training.

🔹Addressed data incompatibility issues between tabular and image datasets by ensuring proper feature extraction and fusion techniques, enabling seamless integration for hybrid models.

🔹Resolved training challenges by implementing early stopping and model checkpoints, preventing overfitting and ensuring model stability during extended training periods.

🔹Optimized model performance by troubleshooting and correcting dimensionality mismatches between inputs, refining the architecture of late fusion models to improve accuracy across all datasets.
