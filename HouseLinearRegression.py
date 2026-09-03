import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


df = pd.read_csv('house_price_regression_dataset.csv')
print(df.shape)

print('Missing values')
print(df.isnull())

X = df.drop('House_Price',axis=1)
y = df['House_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print("Traning:",X_train.shape)
print('testing:',X_test.shape)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# print(y_pred)

new_house = pd.DataFrame({
    "Square_Footage":[5000],
    "Num_Bedrooms":[3],
    "Num_Bathrooms":[5],
    "Year_Built":[2020],
    "Lot_Size":[5000],
    "Garage_Size":[2],
    "Neighborhood_Quality":[8]
})
Perdicted_price = model.predict(new_house)
print("predicted House price:",Perdicted_price[0])

plt.scatter(X_train["Square_Footage"], y_train)
plt.plot(X_train["Square_Footage"], model.predict(X_train),color="black")
plt.xlabel("Square Footage")
plt.ylabel("House Price")
plt.title("Square Footage vs House Price")
plt.show()