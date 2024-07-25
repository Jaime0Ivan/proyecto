from flask import Flask, request, render_template, jsonify
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from joblib import load

app = Flask(__name__)

# Cargar los modelos y transformadores
ordinal_encoder = load('ordinal_attendance.pkl')
scaler = load('scaler_attendance.pkl')
model = load('attendance_model.pkl')

@app.route('/', methods=['GET'])
def index():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        df = pd.DataFrame(data, index=[0])
        
        categorical_columns = ['Materia', 'DíaSemana', 'Profesor', 'Hora', 'Traslado']
        df_encoded = df.copy()
        df_encoded[categorical_columns] = ordinal_encoder.transform(df[categorical_columns])
        
        df_scaled = scaler.transform(df_encoded)
        
        prediction = model.predict(df_scaled)
        
        return jsonify({'prediction': 'Asistencia' if prediction[0] == 1 else 'No Asistencia'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)