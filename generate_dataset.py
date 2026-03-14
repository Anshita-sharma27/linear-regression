import pandas as pd
import numpy as np

np.random.seed(42)

rows = 1500

experience = np.random.randint(0, 20, rows)
education = np.random.randint(1, 4, rows)  
age = np.random.randint(21, 60, rows)
skills = np.random.randint(1, 10, rows)

salary = (
    experience * 5000 +
    education * 10000 +
    skills * 3000 +
    age * 500 +
    np.random.randint(-5000,5000,rows)
)

data = pd.DataFrame({
    "YearsExperience": experience,
    "EducationLevel": education,
    "Age": age,
    "SkillsScore": skills,
    "Salary": salary
})

data.to_csv("salary_dataset_large.csv", index=False)

print("Dataset Created Successfully!")
print(data.head())