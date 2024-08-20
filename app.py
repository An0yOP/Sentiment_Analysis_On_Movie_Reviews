from flask import Flask, render_template, request, url_for
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
import nltk
#down = nltk.download('stopwords')

app = Flask(__name__)

# Load the model and vectorizer
model_path = r'Model\IMDB_trained_model.pkl'
vectorizer_path = r'Model\vectorizer_model.pkl'

predictor = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))

STOPWORDS = set(stopwords.words("english"))

@app.route("/", methods=["GET", "POST"])
def homePage():
    if request.method == "POST":
        text_input = request.form["txtBox"]
        predicted_sentiment = make_prediction(text_input)
        return render_template('index.html', prediction=predicted_sentiment, text_input=text_input)
    return render_template('index.html')


def make_prediction(text_input):
    port_stem = PorterStemmer()
    
    # Text preprocessing
    stemmed_content = re.sub('[^a-zA-Z]', ' ', text_input)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in STOPWORDS]
    stemmed_content = ' '.join(stemmed_content)

    # Vectorize the input using the loaded vectorizer
    transformed_input = vectorizer.transform([stemmed_content])

    # Make prediction
    prediction = predictor.predict(transformed_input)

    if prediction[0] == 0:
        return '<div class="red-warning">The review is Negative.</div>'
    else:
        return '<div class="green-success">The review is Positive.</div>'

if __name__ == "__main__":
    app.run(debug=True)
