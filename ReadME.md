# 🏠 California Housing Price Prediction

A Machine Learning project that predicts **median house values** using the California Housing dataset.

This project demonstrates an end-to-end Machine Learning workflow using **Python, Pandas, NumPy, Scikit-learn, and Random Forest Regression**. It includes data preprocessing, stratified train-test splitting, feature transformation, model training, model persistence, and inference on new data.

## 📌 Project Overview

The goal of this project is to build a regression model that predicts the `median_house_value` of a California district based on features such as:

* Longitude
* Latitude
* Housing median age
* Total rooms
* Total bedrooms
* Population
* Households
* Median income
* Ocean proximity

The project uses a preprocessing pipeline to handle missing values, scale numerical features, and encode categorical features before training the model.

## 🛠️ Technologies Used

* **Python**
* **Pandas** — Data manipulation
* **NumPy** — Numerical operations
* **Scikit-learn** — Machine Learning and preprocessing
* **Joblib** — Saving and loading trained models
* **Random Forest Regressor** — Prediction model

## 🔄 Machine Learning Workflow

```text
California Housing Dataset
          ↓
     Load Dataset
          ↓
   Create income category
          ↓
Stratified Train/Test Split
          ↓
 Separate Features & Labels
          ↓
   Data Preprocessing
     ↙           ↘
Numerical       Categorical
Features        Features
   ↓                ↓
Median Imputer   One-Hot Encoding
   ↓
Standard Scaling
          ↓
   ColumnTransformer
          ↓
Random Forest Regressor
          ↓
    Trained Model
          ↓
      Prediction
          ↓
     output.csv
```

## 📂 Project Structure

```text
California-Housing-ML/
│
├── main2.py
├── housing.csv
├── input.csv
├── output.csv
├── model.pkl
├── pipeline.pkl
├── README.md
└── .gitignore
```

> `model.pkl`, `pipeline.pkl`, `input.csv`, and `output.csv` are generated during the execution of the project.

## 🧹 Data Preprocessing

### 1. Handling Missing Values

The numerical pipeline uses `SimpleImputer` with the median strategy:

```python
SimpleImputer(strategy="median")
```

This replaces missing numerical values with the median value of the corresponding feature.

### 2. Feature Scaling

Numerical features are standardized using:

```python
StandardScaler()
```

This puts numerical features on a comparable scale.

### 3. Categorical Encoding

The `ocean_proximity` feature is categorical, so it is converted into numerical features using:

```python
OneHotEncoder(handle_unknown="ignore")
```

The `handle_unknown="ignore"` option also allows the pipeline to process previously unseen categories during prediction.

## 🤖 Machine Learning Model

The project uses:

```python
RandomForestRegressor(random_state=42)
```

Random Forest combines multiple decision trees to produce a stronger regression model and is well suited to this type of tabular dataset.

## 💾 Model Persistence

After training, both the model and preprocessing pipeline are saved using Joblib:

```python
joblib.dump(model, "model.pkl")
joblib.dump(pipeline, "pipeline.pkl")
```

This allows the trained model to be reused without retraining every time.

## 🔮 Making Predictions

When `model.pkl` already exists, the program enters inference mode.

It loads:

```python
model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")
```

Then it reads new housing data:

```python
input_data = pd.read_csv("input.csv")
```

The same preprocessing pipeline is applied:

```python
transformed_input = pipeline.transform(input_data)
```

Finally, predictions are generated:

```python
predictions = model.predict(transformed_input)
```

The predicted values are added to the input data and saved as:

```text
output.csv
```

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd California-Housing-ML
```

### 2. Install the required packages

```bash
pip install numpy pandas scikit-learn joblib
```

### 3. Run the program

```bash
python main2.py
```

### 4. Training

If `model.pkl` does not exist, the program will:

1. Load `housing.csv`
2. Create income categories
3. Perform a stratified train-test split
4. Separate features and labels
5. Build the preprocessing pipeline
6. Train the Random Forest model
7. Save the model and pipeline

You should see:

```text
Model is trained Congratss!!
```

### 5. Prediction

After the model has been trained, running the program again will load the saved model and pipeline.

The program will:

1. Read `input.csv`
2. Transform the input data
3. Generate predictions
4. Save the results to `output.csv`

## 📊 Dataset

The project uses the **California Housing dataset**, which contains information about housing districts in California.

The target variable is:

```text
median_house_value
```

The model uses the remaining housing characteristics to predict this value.

## 🧠 Key Concepts Learned

This project helped demonstrate several important Machine Learning concepts:

* Train-test splitting
* Stratified sampling
* Feature engineering
* Numerical preprocessing
* Missing-value imputation
* Feature scaling
* One-hot encoding
* Pipelines
* Column transformers
* Random Forest Regression
* Model serialization
* Inference using a saved model

## 🔧 Future Improvements

Possible improvements for this project include:

* Compare Linear Regression, Decision Tree, and Random Forest models
* Perform cross-validation
* Calculate RMSE and MAE
* Tune Random Forest hyperparameters using `GridSearchCV` or `RandomizedSearchCV`
* Add model evaluation visualizations
* Build a Streamlit web application
* Deploy the model as an API
* Add automated data validation

## 👨‍💻 Author

**Marut Shukla**

B.Tech | Aspiring Data Analyst / Data Scientist

### Skills Demonstrated

`Python` `Pandas` `NumPy` `Scikit-learn` `Machine Learning` `Data Preprocessing` `Random Forest` `GitHub`

---

⭐ If you found this project useful, feel free to explore the repository and connect with me on LinkedIn.
