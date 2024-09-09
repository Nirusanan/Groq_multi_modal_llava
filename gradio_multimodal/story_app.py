import gradio as gr
from groq import Groq
import base64
from io import BytesIO

client = Groq()
llava_model = 'llava-v1.5-7b-4096-preview'
llm_model = 'llama-3.1-70b-versatile'

prompt = '''
Describe this image in more detail.
'''

# Image encoding
def encode_image(image_input):
    buffered = BytesIO()
    image_input.save(buffered, format="JPEG")  # Save the image in-memory as JPEG
    return base64.b64encode(buffered.getvalue()).decode('utf-8')
  

# Image-to-text function 
def image_to_text(image):
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


def short_story_generation(image):
    image_description = image_to_text(image)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a children's book author. Write a short story about the scene depicted in this image or images.",
            },
            {
                "role": "user",
                "content": image_description,
            }
        ],
        model=llm_model
    )
    
    return chat_completion.choices[0].message.content  # Return the generated short story


# Gradio interface
image_input = gr.Image(type="pil", label="Upload an Image")

demo = gr.Interface(
    fn=short_story_generation,  
    inputs=[image_input], 
    outputs="text", 
    title="Generate Short Story based on Image"
)

demo.launch()
