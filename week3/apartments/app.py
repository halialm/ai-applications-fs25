import gradio as gr
import numpy as np
import joblib

# Modell laden
model = joblib.load("apartment_price_model.pkl")

# Vorhersage-Funktion
def predict_price(rooms, area, distance_to_lake, luxurious, temporary, furnished, 
                  luxurios, pool, seesicht, exklusiv, attika, loft):
    try:
        # Checkboxes (True/False) zu 1/0 konvertieren
        categorical_features = [luxurious, temporary, furnished, luxurios, pool, seesicht, exklusiv, attika, loft]
        categorical_features = [1 if feature else 0 for feature in categorical_features]

        # Eingabearray erstellen
        input_data = np.array([[rooms, area, distance_to_lake] + categorical_features])

        prediction = model.predict(input_data)
        return f"Estimated Price: {prediction[0]:,.2f} CHF"
    
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio UI definieren
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown(
        "<h1 style='text-align: center;'>Apartment Price Estimator</h1>"
        "<p style='text-align: center;'>Enter the apartment details and click 'Submit' to get an estimated price prediction.</p>"
    )

    with gr.Row():
        with gr.Column():
            rooms = gr.Number(label="Rooms", value=0)
            area = gr.Number(label="Area (m²)", value=0)
            distance_to_lake = gr.Number(label="Distance to Lake Zurich (km)", value=0)

            gr.Markdown("### Apartment Features")
            luxurious = gr.Checkbox(label="Luxurious")
            temporary = gr.Checkbox(label="Temporary")
            furnished = gr.Checkbox(label="Furnished")
            luxurios = gr.Checkbox(label="Luxurious (LUXURIÖS)")
            pool = gr.Checkbox(label="Pool")
            seesicht = gr.Checkbox(label="Seesicht")
            exklusiv = gr.Checkbox(label="Exklusiv")
            attika = gr.Checkbox(label="Attika")
            loft = gr.Checkbox(label="Loft")

        with gr.Column():
            result = gr.Textbox(label="Predicted Apartment Price")
            gr.Button("Submit").click(
                predict_price, 
                inputs=[rooms, area, distance_to_lake, luxurious, temporary, furnished, luxurios, pool, seesicht, exklusiv, attika, loft], 
                outputs=result
            )

    gr.Markdown(
        "<p style='text-align: center;'>Created with Gradio 🚀</p>"
    )

# Starte Gradio App
app.launch()
