from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load models and forecasts
with open('models/time_series_models.pkl', 'rb') as f:
    models = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/forecast', methods=['POST'])
def forecast():
    # Get the selected model from the user
    selected_model = request.form.get('model')
    
    if selected_model not in models:
        return render_template('index.html', error="Invalid model selection.")
    
    # Get the MSE and forecast image path for the selected model
    mse = models[selected_model]['mse']
    image_path = f'static/{selected_model}_forecast.png'
    
    return render_template('index.html', model=selected_model, mse=mse, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
