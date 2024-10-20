from flask import Flask, render_template, request, send_file
import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import movie_reviews
import matplotlib.pyplot as plt
import io

# Step 1: Data Preparation
nltk.download("movie_reviews")

# Load the dataset
documents = [
    (" ".join(movie_reviews.words(fileid)), category)
    for category in movie_reviews.categories()
    for fileid in movie_reviews.fileids(category)
]

# Convert to DataFrame
df = pd.DataFrame(documents, columns=["review", "sentiment"])

# Step 2: Model Training
vectorizer = CountVectorizer(max_features=2000)
X = vectorizer.fit_transform(df["review"])
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

# Flask Application
app = Flask(__name__)

# Function to predict sentiment for a single review
def predict_sentiment(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)
    return prediction[0]

# Function to predict sentiments for multiple reviews from a text file
def process_file(file_content):
    pos_count = 0
    neg_count = 0
    reviews = file_content.splitlines()

    for review in reviews:
        if review.strip():  # Ignore empty lines
            sentiment = predict_sentiment(review)
            if sentiment == 'pos':
                pos_count += 1
            else:
                neg_count += 1

    return pos_count, neg_count

# Function to generate a vertical bar chart for the ratio of positive to negative reviews
def create_vertical_ratio_chart(pos_count, neg_count):
    labels = ['Positive', 'Negative']
    total = pos_count + neg_count
    pos_ratio = pos_count / total
    neg_ratio = neg_count / total

    # Create the plot
    plt.figure(figsize=(4, 6))
    plt.bar(labels, [pos_ratio, neg_ratio], color=['green', 'red'])
    plt.ylim(0, 1)
    plt.ylabel('Ratio')
    plt.title('Positive vs Negative Review Ratio')

    # Annotate the bars with exact ratio values
    for i, v in enumerate([pos_ratio, neg_ratio]):
        plt.text(i, v + 0.02, f"{v:.2%}", ha='center', fontsize=12)

    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    return buf

# Route for homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle form submission for single review
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['review']
        prediction = predict_sentiment(user_input)
        return render_template('index.html', prediction=prediction, user_input=user_input)

# Route to handle file upload and process reviews in the file
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', file_error="No file part")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', file_error="No selected file")

    if file:
        file_content = file.read().decode("utf-8")
        pos_count, neg_count = process_file(file_content)
        return render_template('index.html', pos_count=pos_count, neg_count=neg_count)

# Route to generate and display the chart
@app.route('/chart')
def chart():
    pos_count = int(request.args.get('pos_count', 0))
    neg_count = int(request.args.get('neg_count', 0))

    buf = create_vertical_ratio_chart(pos_count, neg_count)
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
