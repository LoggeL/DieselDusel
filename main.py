import os
from pydantic import BaseModel, Field
from groq import Groq
from dotenv import load_dotenv
import base64


class FuelStats(BaseModel):
    """Fuel Stats"""

    consumption: float = Field(..., description="Verbrauch")
    km: int = Field(..., description="km")
    trip: str = Field(..., description="Trip km (number on bottom right of the display)")


# Umgebungsvariablen laden
load_dotenv()

# Clients initialisieren
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Path to your image
image_path = "pics/photo_2024-11-04_20-26-23.jpg"


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Getting the base64 string
base64_image = encode_image(image_path)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Fill the given JSON schema using the provided image: {FuelStats.schema()}",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        }
    ],
    temperature=1.0,
    max_tokens=1024,
    top_p=1,
    model="llama-3.2-90b-vision-preview",
    response_format={"type": "json_object"},
)

# Define your desired data structure.
response_text = chat_completion.choices[0].message.content

response = FuelStats.model_validate_json(response_text)

print(response)
