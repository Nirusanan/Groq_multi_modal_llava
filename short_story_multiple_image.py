from groq import Groq
import base64

client = Groq()
llava_model = 'llava-v1.5-7b-4096-preview'
llama31_model = 'llama-3.1-70b-versatile'

# Image encoding
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


# Image to text function
def image_to_text(client, model, base64_image, prompt):
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
        model=model
    )

    return chat_completion.choices[0].message.content


# Short story generation function
def short_story_generation(client, image_description):
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
        model=llama31_model
    )
    
    return chat_completion.choices[0].message.content


# Multiple image processing
image1 = encode_image('images\dog_1.jpg')
image2 = encode_image('images\dog_2.jpg')

prompt = "Describe this image"

image_description1 = image_to_text(client, llava_model, image1, prompt)
image_description2 = image_to_text(client, llava_model, image2, prompt)

print("\n--- Image Description (Image1) ---")
print(image_description1)

print("\n--- Image Description (Image2) ---")
print(image_description2)

combined_image_description = image_description1 + '\n\n' + image_description2

print("\n--- Short Story (Based on Image1 and Image2) ---")
print(short_story_generation(client, combined_image_description))
