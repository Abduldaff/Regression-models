# House Price Prediction

This project is a beginner-friendly collection of machine learning examples for predicting house prices. It uses the same dataset with several regression algorithms so you can compare how different models learn relationships between house features and price.

The code is intended for learning and experimentation, not for making real property valuations.

## Learning Goals

By working through this project, you can practice:

- Loading and inspecting a CSV dataset with pandas
- Separating input features (`X`) from the target value (`y`)
- Splitting data into training and testing sets
- Training regression models with scikit-learn
- Making predictions for a new house
- Comparing actual and predicted prices with plots
- Understanding why feature scaling, model complexity, and hyperparameters matter

## Project Files

| File | Model | Main idea |
| --- | --- | --- |
| `HouseLinearRegression.py` | Linear regression | Shows a simple linear relationship between square footage and price. |
| `MultiLinearRegression.py` | Multiple linear regression | Uses all seven house features to predict price. |
| `PolynomialRegression.py` | Degree-2 polynomial regression | Adds squared and interaction features to model nonlinear relationships. |
| `DecisionTreeRegression.py` | Decision tree regression | Learns rule-based splits in the feature data. |
| `SupportVectorRegression.py` | Support vector regression | Uses an RBF kernel and standardizes the input features first. |
| `house_price_regression_dataset.csv` | Dataset | Contains the house features and prices used by every script. |

## Dataset

The dataset contains these columns:

- `Square_Footage`
- `Num_Bedrooms`
- `Num_Bathrooms`
- `Year_Built`
- `Lot_Size`
- `Garage_Size`
- `Neighborhood_Quality`
- `House_Price` (the value to predict)

The first seven columns are input features. `House_Price` is the target column.

## Requirements

- Python 3.9 or newer
- pandas
- matplotlib
- scikit-learn

Install the required packages from a terminal:

```bash
python -m pip install pandas matplotlib scikit-learn
```

## How to Run

Open a terminal in this project folder, then run one script at a time:

```bash
python HouseLinearRegression.py
python MultiLinearRegression.py
python PolynomialRegression.py
python DecisionTreeRegression.py
python SupportVectorRegression.py
```

Each script loads `house_price_regression_dataset.csv`, trains a model, prints a predicted price for an example house, and opens a Matplotlib chart. Close the chart window before running another script.

Run the commands from the project folder because the scripts use a relative path to find the CSV file.

## Common Workflow

Every example follows this basic machine learning process:

1. Load the dataset.
2. Store the feature columns in `X` and the price column in `y`.
3. Split the data into training and testing sets.
4. Fit a regression model using the training data.
5. Predict prices for the test data and for a new house.
6. Visualize the predictions.

The examples use `test_size=0.2` and `random_state=42`, which means 80% of the rows are used for training and 20% for testing. The fixed random state makes the split repeatable.

## Suggested Learning Order

1. Start with `HouseLinearRegression.py` to understand the basic structure of a regression program.
2. Read `MultiLinearRegression.py` and see how multiple features are used together.
3. Try `PolynomialRegression.py` to see how a model can represent more complex relationships.
4. Study `DecisionTreeRegression.py` as a different, rule-based approach.
5. Finish with `SupportVectorRegression.py` and examine why `StandardScaler` is useful for support vector models.

## Ideas for Further Practice

- Add MAE, RMSE, and R-squared evaluation metrics.
- Compare all model scores in one table.
- Change the decision tree `max_depth` and observe the result.
- Experiment with the polynomial degree.
- Try different SVR values for `C` and `epsilon`.
- Sort the square-footage values before drawing the simple regression line.
- Add a `requirements.txt` file and an `if __name__ == "__main__":` entry point to each script.

## Note About Results

The sample predictions and plots are useful for understanding the algorithms, but a good-looking prediction does not guarantee that a model is accurate. Always evaluate a model with suitable metrics and real-world data before using it for decisions.