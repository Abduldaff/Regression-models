import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.read_csv('house_price_regression_dataset.csv')

X = df.drop('House_Price', axis = 1)
y = df['House_Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
model =LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

new_house = pd.DataFrame({
    "Square_Footage": [8000],
    "Num_Bedrooms": [5],
    "Num_Bathrooms": [5],
    "Year_Built": [2026],
    "Lot_Size": [2000],
    "Garage_Size": [5],
    "Neighborhood_Quality": [5]
})
predicted_price = model.predict(new_house)
print("Predicted House Price:", predicted_price[0])

plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()],color="red",linestyle="--")
plt.xlabel("Actual House price")
plt.ylabel('predicted house price')
plt.title('Multi linear regression')
plt.show()