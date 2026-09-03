import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

df = pd.read_csv('house_price_regression_dataset.csv')

X = df.drop("House_Price", axis = 1)
y = df['House_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = Pipeline([
    ('poly',PolynomialFeatures(degree = 2)),
    ('linear',LinearRegression())
])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
new_house = pd.DataFrame({
    "Square_Footage": [3500],
    "Num_Bedrooms": [2],
    "Num_Bathrooms": [1],
    "Year_Built": [2022],
    "Lot_Size": [2000],
    "Garage_Size": [0],
    "Neighborhood_Quality": [5]
})
predicted_price = model.predict(new_house)
print("Predicted House Price:", predicted_price[0])

plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()],color="red",linestyle="--")
plt.xlabel('actual house price')
plt.ylabel('predicted house price')
plt.title('ploynomial regression')
plt.show()
