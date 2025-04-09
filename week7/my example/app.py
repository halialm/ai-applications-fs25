import gradio as gr
from transformers import pipeline

# Modelle laden
vit_classifier = pipeline("image-classification", model="halimalm/vit-base-oxford-iiit-pets")
clip_detector = pipeline(model="openai/clip-vit-large-patch14", task="zero-shot-image-classification")

# Alle Labels aus dem Oxford Pets Dataset
labels_oxford_pets = [
    'Siamese', 'Birman', 'shiba inu', 'staffordshire bull terrier', 'basset hound', 'Bombay', 'japanese chin',
    'chihuahua', 'german shorthaired', 'pomeranian', 'beagle', 'english cocker spaniel', 'american pit bull terrier',
    'Ragdoll', 'Persian', 'Egyptian Mau', 'miniature pinscher', 'Sphynx', 'Maine Coon', 'keeshond', 'yorkshire terrier',
    'havanese', 'leonberger', 'wheaten terrier', 'american bulldog', 'english setter', 'boxer', 'newfoundland', 'Bengal',
    'samoyed', 'British Shorthair', 'great pyrenees', 'Abyssinian', 'pug', 'saint bernard', 'Russian Blue', 'scottish terrier'
]

# Klassifikationsfunktion
def classify_pet(image):
    vit_results = vit_classifier(image)
    vit_output = {result['label']: round(result['score'], 4) for result in vit_results}
    
    clip_results = clip_detector(image, candidate_labels=labels_oxford_pets)
    clip_output = {result['label']: round(result['score'], 4) for result in clip_results}
    
    return {
        "Transfer Learning (ViT)": vit_output,
        "Zero-Shot (CLIP)": clip_output
    }

# Beispielbilder
example_images = [
    ["example_images/cat.jpg"],
    ["example_images/dog1.jpeg"],
    ["example_images/dog2.jpeg"]
]

# Gradio Interface
iface = gr.Interface(
    fn=classify_pet,
    inputs=gr.Image(type="filepath"),
    outputs=gr.JSON(),
    title="Oxford Pets: ViT vs. CLIP",
    description="Upload an image of a pet and compare predictions from a trained ViT model and a zero-shot CLIP model.",
    examples=example_images
)

iface.launch()
