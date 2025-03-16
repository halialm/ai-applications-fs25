import gradio as gr
import numpy as np
import joblib

# Modell laden
model = joblib.load("apartment_price_model.pkl")

# Vorhersage-Funktion
def predict_price(rooms, area, distance_to_lake, room_per_m2, price_per_m2, luxurious, temporary, furnished, 
                  luxurios, pool, seesicht, exklusiv, attika, loft):
    try:
        # Checkboxes (True/False) zu 1/0 konvertieren
        categorical_features = [luxurious, temporary, furnished, luxurios, pool, seesicht, exklusiv, attika, loft]
        categorical_features = [1 if feature else 0 for feature in categorical_features]

        # Eingabearray erstellen
        input_data = np.array([[distance_to_lake, rooms, area, room_per_m2, price_per_m2] + categorical_features])

        prediction = model.predict(input_data)
        return f"Estimated Price: {prediction[0]:,.2f} CHF"
    
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio UI definieren
iface = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Number(label="Rooms"),
        gr.Number(label="Area (m²)"),
        gr.Number(label="Distance to Lake Zurich (km)"),
        gr.Number(label="Room per m²"),
        gr.Number(label="Price per m²"),
        gr.Checkbox(label="Luxurious"),
        gr.Checkbox(label="Temporary"),
        gr.Checkbox(label="Furnished"),
        gr.Checkbox(label="Luxurious (LUXURIÖS)"),
        gr.Checkbox(label="Pool"),
        gr.Checkbox(label="Seesicht"),
        gr.Checkbox(label="Exklusiv"),
        gr.Checkbox(label="Attika"),
        gr.Checkbox(label="Loft")
    ],
    outputs=gr.Textbox(label="Predicted Apartment Price"),
    title="Apartment Price Estimator",
    description="Enter the apartment details and click 'Submit' to get an estimated price prediction.",
    live=False
)

# Starte Gradio App
iface.launch()
