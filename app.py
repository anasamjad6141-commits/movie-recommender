from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load dataset
df = pd.read_csv("movies.csv")

# Encode genre
le = LabelEncoder()
df["genre_encoded"] = le.fit_transform(df["genre"])

# Features
X = df[["genre_encoded","rating","duration","year"]]
y = df["liked"]

# Train model
model = LogisticRegression()
model.fit(X,y)


@app.route("/", methods=["GET","POST"])
def index():

    recommendations = []

    if request.method == "POST":

        movie_name = request.form["movie"]

        if movie_name in df["title"].values:

            movie = df[df["title"] == movie_name].iloc[0]

            df["probability"] = model.predict_proba(X)[:,1]

            rec = df[
                (df["genre"] == movie["genre"]) &
                (df["title"] != movie_name)
            ].sort_values(by="probability",ascending=False).head(8)

            recommendations = rec.to_dict(orient="records")

    return render_template(
        "index.html",
        movies=df["title"].values,
        recommendations=recommendations
    )


if __name__ == "__main__":
    app.run(debug=True)