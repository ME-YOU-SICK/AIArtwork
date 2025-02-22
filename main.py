import os
import json
import re
import requests
import dotenv
import logging
import uuid
import shutil
import time
from gradio_client import Client
from PIL import Image

# Load environment variables
dotenv.load_dotenv()

# Configuration from environment variables
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
HUGGINGFACE_FLUX_API = os.getenv("HUGGINGFACE_FLUX_API")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Check for required environment variables
required_vars = ["OPENROUTER_API_KEY", "HUGGINGFACE_FLUX_API", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]
for var in required_vars:
    if not os.getenv(var):
        logging.error(f"Missing environment variable: {var}")
        exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def sanitize_filename(text, max_length=50):
    """
    Sanitize text to be safe for use as a filename.
    - Removes characters not allowed in filenames.
    - Replaces whitespace with underscores.
    - Limits the length to max_length characters.
    """
    sanitized = re.sub(r'[\\/*?:"<>|]', "", text)
    sanitized = re.sub(r'\s+', '_', sanitized)
    return sanitized[:max_length]


# DeepSeek: Generate a concise image idea
def generate_image_idea():
    payload = {
        "model": "deepseek/deepseek-r1-distill-llama-70b:free",
        "messages": [{"role": "user", "content": "Give me a very short AI image idea in 5-10 words."}]
    }
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        idea = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        return idea if idea else None
    except requests.RequestException as e:
        logging.error(f"Error generating image idea: {e}")
        return None


# DeepSeek: Generate a detailed prompt based on the idea
def generate_detailed_prompt(idea):
    payload = {
        "model": "deepseek/deepseek-r1-distill-llama-70b:free",
        "messages": [{"role": "user", "content": f"Create a short AI image prompt (under 50 words) for: {idea}"}]
    }
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        prompt = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        return prompt if prompt else None
    except requests.RequestException as e:
        logging.error(f"Error generating detailed prompt: {e}")
        return None


# FLUX.1: Generate AI Image using the prompt
def generate_ai_image(prompt):
    client = Client(HUGGINGFACE_FLUX_API)
    try:
        result = client.predict(
            prompt=prompt,
            seed=0,
            randomize_seed=True,
            width=1024,
            height=1024,
            guidance_scale=3.5,
            num_inference_steps=28,
            api_name="/infer"
        )
        if isinstance(result, tuple) and len(result) >= 1 and isinstance(result[0], str):
            local_path = result[0]
            if os.path.exists(local_path):
                return local_path
            logging.error(f"Local file path {local_path} does not exist.")
            return None
        elif isinstance(result, list) and result and isinstance(result[0], dict) and "url" in result[0]:
            return result[0]["url"]
        logging.error(f"Unexpected response format from FLUX.1: {result}")
        return None
    except Exception as e:
        logging.error(f"Error generating AI image: {e}")
        return None


# Download or Copy Image with a specific output filename and convert to JPEG
def download_image(image_path_or_url, output_filename):
    try:
        if image_path_or_url.startswith("http"):
            response = requests.get(image_path_or_url, timeout=30)
            response.raise_for_status()
            with open(output_filename, "wb") as file:
                file.write(response.content)
        elif os.path.exists(image_path_or_url):
            shutil.copy(image_path_or_url, output_filename)
        else:
            logging.error(f"Invalid image path or URL: {image_path_or_url}")
            return None

        # Convert image to JPEG (if not already) using Pillow
        img = Image.open(output_filename).convert("RGB")
        img.save(output_filename, "JPEG")
        return output_filename
    except (requests.RequestException, shutil.Error, IOError) as e:
        logging.error(f"Error downloading, copying, or converting image: {e}")
        return None


def send_telegram_message(image_path, caption):
    """
    Send the generated image and caption to the specified Telegram chat using the bot.
    """
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        with open(image_path, "rb") as photo:
            data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
            response = requests.post(url, data=data, files={"photo": photo})
        if response.status_code != 200:
            logging.error(f"Failed to send Telegram message: {response.text}")
        else:
            logging.info("Telegram message sent successfully.")
    except Exception as e:
        logging.error(f"Error sending Telegram message: {e}")


def main_workflow():
    logging.info("Starting AI image generation process...")

    idea = generate_image_idea()
    if not idea:
        logging.error("Aborting due to failure in generating image idea.")
        return
    logging.info(f"Generated idea: {idea}")

    detailed_prompt = generate_detailed_prompt(idea)
    if not detailed_prompt:
        logging.error("Aborting due to failure in generating detailed prompt.")
        return
    logging.info(f"Generated prompt: {detailed_prompt}")

    image_path_or_url = generate_ai_image(detailed_prompt)
    if not image_path_or_url:
        logging.error("Aborting due to failure in generating AI image.")
        return
    logging.info(f"Generated image path or URL: {image_path_or_url}")

    # Construct caption and output filename
    caption = f"Generated by AI: {detailed_prompt} #AIArt #AIGenerated"
    sanitized_caption = sanitize_filename(caption)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_filename = f"{timestamp}_{sanitized_caption}.jpg"

    image_path = download_image(image_path_or_url, output_filename)
    if not image_path:
        logging.error("Aborting due to failure in downloading or copying image.")
        return

    logging.info(f"Image saved as: {image_path}")
    logging.info(f"Caption: {caption}")

    # Send the image and caption to Telegram
    send_telegram_message(image_path, caption)
    logging.info("Process completed successfully.")


if __name__ == "__main__":
    while True:
        main_workflow()
        logging.info("Waiting 20 minutes before next run...")
        time.sleep(20 * 60)  # 20 minutes in seconds
