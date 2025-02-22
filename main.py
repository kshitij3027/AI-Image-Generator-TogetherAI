import os
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from termcolor import colored
from together import Together
import openai
import base64
from PIL import Image
import io
import time

# Constants
MODEL_NAME = "black-forest-labs/FLUX.1-schnell-Free"
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 768
STEPS = 4
GROQ_MODEL = "llama-3.3-70b-versatile"

# Initialize FastAPI app
app = FastAPI(title="Image Generation App")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize templates
templates = Jinja2Templates(directory="templates")

# Initialize Together client with explicit environment variable
try:
    client = Together(api_key=os.environ["TOGETHER_API_KEY"])
    print(colored("✓ Together client initialized successfully", "green"))
except KeyError:
    print(colored("✗ TOGETHER_API_KEY environment variable not found", "red"))
except Exception as e:
    print(colored(f"✗ Error initializing Together client: {str(e)}", "red"))

# Initialize Groq client with explicit environment variable
try:
    groq_client = openai.OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ["GROQ_API_KEY"]
    )
    print(colored("✓ Groq client initialized successfully", "green"))
except KeyError:
    print(colored("✗ GROQ_API_KEY environment variable not found", "red"))
except Exception as e:
    print(colored(f"✗ Error initializing Groq client: {str(e)}", "red"))

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "title": "AI Image Generator"}
    )

async def enhance_prompt(prompt: str) -> str:
    try:
        print(colored(f"Enhancing prompt with Groq: {prompt}", "cyan"))
        response = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at crafting detailed, descriptive prompts for image generation. Convert user inputs into vivid, detailed prompts that will result in high-quality, artistic images. Focus on adding artistic style, lighting, mood, and composition details. Keep the enhanced prompt concise but descriptive."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=GROQ_MODEL,
            temperature=0.7,
            max_completion_tokens=150,
            top_p=1,
        )
        enhanced_prompt = response.choices[0].message.content
        print(colored(f"Enhanced prompt: {enhanced_prompt}", "green"))
        return enhanced_prompt
    except Exception as e:
        print(colored(f"Error enhancing prompt: {str(e)}", "red"))
        return prompt  # Return original prompt if enhancement fails

@app.post("/generate")
async def generate_image(request: Request, prompt: str = Form(...)):
    try:
        # Check if prompt enhancement is requested
        should_enhance = request.headers.get('X-Enhance-Prompt', 'true').lower() == 'true'
        
        # Enhance the prompt using Groq if requested
        enhanced_prompt = await enhance_prompt(prompt) if should_enhance else prompt
        
        print(colored(f"Generating image for {'enhanced ' if should_enhance else ''}prompt: {enhanced_prompt}", "cyan"))
        start_time = time.time()
        
        response = client.images.generate(
            prompt=enhanced_prompt,
            model=MODEL_NAME,
            width=IMAGE_WIDTH,
            height=IMAGE_HEIGHT,
            steps=STEPS,
            n=1,
            response_format="b64_json"
        )
        
        end_time = time.time()
        generation_time = end_time - start_time
        print(colored(f"Image generated in {generation_time:.2f} seconds", "green"))
        
        image_data = response.data[0].b64_json
        return {
            "success": True, 
            "image": image_data, 
            "generation_time": f"{generation_time:.2f}",
            "enhanced_prompt": enhanced_prompt if should_enhance else None
        }
    
    except Exception as e:
        print(colored(f"Error generating image: {str(e)}", "red"))
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
