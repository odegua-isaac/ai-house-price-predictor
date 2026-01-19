import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("data.csv")

# Select features and target
X = data[["size", "bedrooms"]]
y = data["price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# User input
size = int(input("Enter house size (sqm): "))
bedrooms = int(input("Enter number of bedrooms: "))

# Predict price
prediction = model.predict([[size, bedrooms]])
print(f"Estimated House Price: ${prediction[0]:,.2f}")
