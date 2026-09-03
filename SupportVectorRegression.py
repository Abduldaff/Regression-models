import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline


df = pd.read_csv("house_price_regression_dataset.csv")

X = df.drop("House_Price", axis=1)
y = df["House_Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
model = Pipeline([("scaler", StandardScaler()),("svr", SVR(kernel="rbf", C=100000, epsilon=0.1))])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

new_house = pd.DataFrame({
    "Square_Footage": [5000],
    "Num_Bedrooms": [3],
    "Num_Bathrooms": [5],
    "Year_Built": [2020],
    "Lot_Size": [5000],
    "Garage_Size": [2],
    "Neighborhood_Quality": [8]
})
predicted_price = model.predict(new_house)
print("Predicted House Price:", predicted_price[0])

plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()],color="red",linestyle="--")
plt.xlabel("actual house price")
plt.ylabel("Predicted house price")
plt.title("support vector regression")
plt.show()