from flask import Flask, request, render_template, jsonify
import pickle

app = Flask(__name__)

# Load the model and vectorizer from the .pkl file
with open('models/vibechecker_model.pkl', 'rb') as f:
    model, vectorizer = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        # Transform the input review using the loaded vectorizer
        transformed_review = vectorizer.transform([review])
        # Predict sentiment
        sentiment = model.predict(transformed_review)[0]
        # Convert numerical output to sentiment label
        sentiment_label = 'Positive' if sentiment == 1 else 'Negative'
        return render_template('index.html', prediction_text=f'Sentiment: {sentiment_label}')
    
# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
