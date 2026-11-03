
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==========================================
# STEP 1: GENERATE AND EXPLORE
# ==========================================
print("1. Reading and merging data...")
train_df = pd.read_csv(r"C:\Users\NC\Downloads\train.csv", low_memory=False)
store_df = pd.read_csv(r"C:\Users\NC\Downloads\store.csv")

# Merge sales data with store information
df = train_df.merge(store_df, on="Store", how="left")

# Extract date features
df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day

# Convert StateHoliday to string to avoid One-Hot Encoding errors
df["StateHoliday"] = df["StateHoliday"].astype(str)

print("Missing values before processing:")
print(df.isna().sum())


# ==========================================
# STEP 2: SPLIT FIRST
# ==========================================
print("\n2. Splitting data chronologically...")
# Note: Time-series data must be sorted by time without shuffling
df = df.sort_values("Date")

# Remove closed days to avoid noise in the model
df = df[df["Open"] == 1]

# 80/20 time-based train-test split
cut_idx = int(len(df) * 0.8)
train_set = df.iloc[:cut_idx].copy()
test_set = df.iloc[cut_idx:].copy()

# Define features and target variable ('Sales')
# Drop 'Customers' as it is unknown at prediction time
drop_cols = ["Sales", "Customers", "Date", "Open"]
X_train = train_set.drop(columns=drop_cols)
y_train = train_set["Sales"]

X_test = test_set.drop(columns=drop_cols)
y_test = test_set["Sales"]


# ==========================================
# STEP 3: TRANSFORMATION PIPELINE
# ==========================================
print("\n3. Building Transformation Pipeline...")

# Define numeric and categorical features
num_features = ["CompetitionDistance", "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
                "Promo2SinceWeek", "Promo2SinceYear", "Promo", "Promo2", "SchoolHoliday",
                "DayOfWeek", "Year", "Month", "Day"]
cat_features = ["StoreType", "Assortment", "StateHoliday", "PromoInterval"]

# Numeric pipeline: Impute with median and scale
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical pipeline: Impute missing and one-hot encode
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# Combine pipelines using ColumnTransformer
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features)
])

# Fit the preprocessor ONLY on the training set to prevent data leakage
print("Fitting pipeline on training data...")
X_train_prepared = preprocessor.fit_transform(X_train)

# Transform the test set
print("Transforming test data...")
X_test_prepared = preprocessor.transform(X_test)

print(f"\nData preparation completed! Final Training set shape: {X_train_prepared.shape}")

# STEP 4: TRAIN & EVALUATE
# ==========================================
print("\n4. Training Decision Tree model...")
# Initialize model with max_depth to prevent overfitting
tree_model = DecisionTreeRegressor(random_state=42, max_depth=15)

# Train the model on historical data
tree_model.fit(X_train_prepared, y_train)

print("Predicting sales for the Test set...")
predictions = tree_model.predict(X_test_prepared)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(f"\n✅ FINAL EVALUATION RESULTS:")
print(f"- Training set shape: {X_train_prepared.shape}")
print(f"- MAE (Mean Absolute Error): {mae:,.0f} Euros")
print(f"- RMSE (Root Mean Squared Error): {rmse:,.0f} Euros")