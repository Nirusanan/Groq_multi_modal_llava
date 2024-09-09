import gradio as gr
from groq import Groq
import base64
from io import BytesIO

client = Groq()
llava_model = 'llava-v1.5-7b-4096-preview'

# Image encoding
def encode_image(image_input):
    buffered = BytesIO()
    image_input.save(buffered, format="JPEG")  
    return base64.b64encode(buffered.getvalue()).decode('utf-8')
  

# Image-to-text function 
def image_to_text(image, prompt):
    base64_image = encode_image(image)  
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        model=llava_model
    )

    return chat_completion.choices[0].message.content  


# Gradio interface
image_input = gr.Image(type="pil", label="Upload an Image")
prompt_input = gr.Textbox(label="Visual Question Answering")

demo = gr.Interface(
    fn=image_to_text,  
    inputs=[image_input, prompt_input], 
    outputs="text", 
    title="Visual Question Answering"
)

demo.launch()
