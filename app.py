from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

data = pd.read_csv("salary_dataset_large.csv")

X = data[["YearsExperience","EducationLevel","Age","SkillsScore"]]
y = data["Salary"]

model = LinearRegression()
model.fit(X,y)

@app.route("/", methods=["GET","POST"])
def home():

    prediction = None

    if request.method == "POST":

        exp = float(request.form["experience"])
        edu = float(request.form["education"])
        age = float(request.form["age"])
        skill = float(request.form["skills"])

        result = model.predict([[exp,edu,age,skill]])
        prediction = round(result[0],2)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run()