
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"


def build_pipelline(num_attribs, cat_attribs):
    # Numerical pipeline
    num_pipeline = Pipeline([
        ("impter", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Categorical pipeline
    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    # Full preprocessing pipeline
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
    ])

    return full_pipeline


if not os.path.exists(MODEL_FILE) or not os.path.exists(PIPELINE_FILE):

    # Load data
    housing = pd.read_csv("housing.csv")

    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5]
    )

    # Stratified split
    split = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )

    train_index, test_index = next(
        split.split(housing, housing["income_cat"])
    )

    strat_train_set = housing.iloc[train_index].drop(
        "income_cat", axis=1
    )
    strat_test_set = housing.iloc[test_index].drop(
        "income_cat", axis=1
    )

    # Keep your existing CSV workflow
    strat_test_set.to_csv("input.csv", index=False)
    

    # Features and labels
    housing_labels = strat_train_set["median_house_value"].copy()
    housing_features = strat_train_set.drop(
        "median_house_value", axis=1
    )

    num_attribs = housing_features.drop(
        "ocean_proximity", axis=1
    ).columns.tolist()

    cat_attribs = ["ocean_proximity"]

    # Preprocessing
    preprocessing = build_pipelline(
        num_attribs, cat_attribs
    )

    # Complete pipeline: preprocessing + model
    full_model_pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("model", RandomForestRegressor(random_state=42))
    ])

    # Hyperparameter search space
    param_distribs = {
        "model__n_estimators": [50, 100, 150, 200],
        "model__max_depth": [10, 20, 30, None],
        "model__min_samples_split": [2, 5, 10],
        "model__max_features": [0.5, 1.0]
    }

    rnd_search = RandomizedSearchCV(
        estimator=full_model_pipeline,
        param_distributions=param_distribs,
        n_iter=10,
        cv=3,
        scoring="neg_root_mean_squared_error",
        random_state=42,
        n_jobs=-1,
        verbose=2
    )

    # Fit and tune
    rnd_search.fit(housing_features, housing_labels)

    best_pipeline = rnd_search.best_estimator_

    print("\nBest Parameters:")
    print(rnd_search.best_params_)

    print("\nBest CV RMSE:")
    print(-rnd_search.best_score_)

    # Save model and preprocessing separately for compatibility
    model = best_pipeline.named_steps["model"]
    preprocessing = best_pipeline.named_steps["preprocessing"]

    joblib.dump(model, MODEL_FILE)
    joblib.dump(preprocessing, PIPELINE_FILE)

    print("\nModel and pipeline saved successfully.")

else:
    # Load saved model and preprocessing
    model = joblib.load(MODEL_FILE)
    preprocessing = joblib.load(PIPELINE_FILE)

    # Load held-out test data
    test_data = pd.read_csv("input.csv")

    X_test = test_data.drop("median_house_value", axis=1)
    y_test = test_data["median_house_value"]

    # Preprocess test data
    X_test_prepared = preprocessing.transform(X_test)

    # Predictions
    predictions = model.predict(X_test_prepared)

    # Evaluation
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("\n--- Test Set Evaluation ---")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2 Score:", r2)

    # Save results
    results = X_test.copy()
    results["Actual House Value"] = y_test.to_numpy()
    results["Predicted House Value"] = predictions

    results.to_csv("output.csv", index=False)

    print("\nTest evaluation complete.")
    print("Results saved to output.csv")

