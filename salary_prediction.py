import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("salary_dataset_large.csv")

print("First 5 rows")
print(data.head())

# Independent variables
X = data[["YearsExperience", "EducationLevel", "Age", "SkillsScore"]]

# Target variable
y = data["Salary"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
print("\nModel Performance")
print("R2 Score:", r2_score(y_test, predictions))
print("Mean Squared Error:", mean_squared_error(y_test, predictions))

# Example prediction
example = [[5, 2, 30, 7]]

predicted_salary = model.predict(example)

print("\nExample Prediction")
print("Predicted Salary:", predicted_salary[0])

# Visualization
plt.scatter(y_test, predictions)
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted Salary")
plt.show()