import os
import json
import requests
import dotenv
import logging
import uuid
import shutil
import time
from gradio_client import Client
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# Load environment variables
dotenv.load_dotenv()

# Configuration from environment variables
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
INSTAGRAM_USERNAME = os.getenv("INSTAGRAM_USERNAME")
INSTAGRAM_PASSWORD = os.getenv("INSTAGRAM_PASSWORD")
HUGGINGFACE_FLUX_API = os.getenv("HUGGINGFACE_FLUX_API")

# Check for required environment variables
required_vars = ["OPENROUTER_API_KEY", "INSTAGRAM_USERNAME", "INSTAGRAM_PASSWORD", "HUGGINGFACE_FLUX_API"]
for var in required_vars:
    if not os.getenv(var):
        logging.error(f"Missing environment variable: {var}")
        exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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

# Download or Copy Image with a unique filename and convert to JPEG
def download_image(image_path_or_url):
    filename = f"generated_image_{uuid.uuid4().hex}.jpg"
    try:
        if image_path_or_url.startswith("http"):
            response = requests.get(image_path_or_url, timeout=30)
            response.raise_for_status()
            with open(filename, "wb") as file:
                file.write(response.content)
        elif os.path.exists(image_path_or_url):
            shutil.copy(image_path_or_url, filename)
        else:
            logging.error(f"Invalid image path or URL: {image_path_or_url}")
            return None
        img = Image.open(filename).convert("RGB")
        img.save(filename, "JPEG")
        return filename
    except (requests.RequestException, shutil.Error, IOError) as e:
        logging.error(f"Error downloading, copying, or converting image: {e}")
        return None

# Instagram Bot using Selenium
class InstagramBot:
    def __init__(self):
        self.driver = None
        self.wait = None
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def init_driver(self):
        options = webdriver.ChromeOptions()
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36")
        # Uncomment below line to run in headless mode
        # options.add_argument('--headless')
        self.driver = webdriver.Chrome(options=options)
        self.wait = WebDriverWait(self.driver, 30)  # Increased timeout to 30s
        self.driver.implicitly_wait(10)

    def handle_popup(self, xpath, description):
        """Generic method to handle popups with retry logic"""
        for _ in range(2):  # Retry twice
            try:
                button = self.wait.until(EC.element_to_be_clickable((By.XPATH, xpath)))
                button.click()
                time.sleep(2)
                logging.info(f"{description} popup handled")
                return True
            except TimeoutException:
                logging.info(f"No {description} popup found on attempt {_+1}")
                time.sleep(2)
        return False

    def login(self):
        try:
            self.driver.get('https://www.instagram.com')
            time.sleep(5)
            self.handle_popup(
                '//button[contains(text(), "Allow") or contains(text(), "Accept")]',
                "Cookie consent"
            )
            username_input = self.wait.until(EC.presence_of_element_located(
                (By.CSS_SELECTOR, "input[name='username']")))
            username_input.send_keys(INSTAGRAM_USERNAME)
            password_input = self.wait.until(EC.presence_of_element_located(
                (By.CSS_SELECTOR, "input[name='password']")))
            password_input.send_keys(INSTAGRAM_PASSWORD)
            login_button = self.wait.until(EC.element_to_be_clickable(
                (By.CSS_SELECTOR, "button[type='submit']")))
            login_button.click()
            time.sleep(5)
            self.handle_popup(
                '//button[contains(text(), "Not Now") or contains(text(), "Later")]',
                "Save Login Info"
            )
            self.handle_popup(
                '//button[contains(text(), "Not Now") or contains(text(), "Turn Off")]',
                "Notifications"
            )
            logging.info("Successfully logged in to Instagram")
            return True
        except Exception as e:
            logging.error(f"Failed to login: {str(e)}")
            return False

    def post_image(self, image_path, caption):
        try:
            # Step 1: Click "New Post" button
            logging.info("Clicking 'New Post' button")
            new_post_button = self.wait.until(EC.element_to_be_clickable(
                (By.XPATH, "//*[contains(@aria-label, 'New post') or contains(text(), 'Create')]")))
            new_post_button.click()
            time.sleep(5)  # Increased wait for modal load

            # Step 2: Upload image
            logging.info("Uploading image")
            file_input = self.wait.until(EC.presence_of_element_located(
                (By.XPATH, "//input[@type='file' and @accept='image/jpeg,image/png,video/mp4,video/quicktime']")))
            file_input.send_keys(os.path.abspath(image_path))
            time.sleep(5)  # Wait for upload to process

            # Step 3: Click Next (crop screen)
            logging.info("Clicking 'Next' on crop screen")
            next_button = self.wait.until(EC.element_to_be_clickable(
                (By.XPATH, "//button[normalize-space()='Next']")))
            next_button.click()
            time.sleep(3)

            # Step 4: Click Next (filters screen)
            logging.info("Clicking 'Next' on filters screen")
            next_button = self.wait.until(EC.element_to_be_clickable(
                (By.XPATH, "//button[normalize-space()='Next']")))
            next_button.click()
            time.sleep(3)

            # Step 5: Enter caption
            logging.info("Entering caption")
            caption_input = self.wait.until(EC.presence_of_element_located(
                (By.XPATH, "//textarea[@aria-label='Write a caption...']")))
            caption_input.send_keys(caption)
            time.sleep(3)

            # Step 6: Click Share
            logging.info("Clicking 'Share' button")
            share_button = self.wait.until(EC.element_to_be_clickable(
                (By.XPATH, "//button[normalize-space()='Share']")))
            share_button.click()
            time.sleep(10)  # Increased wait for post to complete

            # Verify post success
            self.wait.until(EC.invisibility_of_element_located(
                (By.XPATH, "//button[normalize-space()='Share']")))
            logging.info("Successfully posted image to Instagram")
            return True

        except TimeoutException as e:
            logging.error(f"Timeout waiting for element: {str(e)}")
            return False
        except NoSuchElementException as e:
            logging.error(f"Element not found: {str(e)}")
            return False
        except Exception as e:
            logging.error(f"Failed to post image: {str(e)}")
            return False

    def close(self):
        if self.driver:
            self.driver.quit()

def post_to_instagram(image_path, caption):
    bot = InstagramBot()
    success = False
    try:
        bot.init_driver()
        if bot.login():
            success = bot.post_image(image_path, caption)
    except Exception as e:
        logging.error(f"Error in Instagram automation: {str(e)}")
    finally:
        bot.close()
    return success

# Main Workflow
if __name__ == "__main__":
    logging.info("Starting AI image generation and posting process...")
    idea = generate_image_idea()
    if not idea:
        logging.error("Aborting due to failure in generating image idea.")
        exit(1)
    logging.info(f"Generated idea: {idea}")
    detailed_prompt = generate_detailed_prompt(idea)
    if not detailed_prompt:
        logging.error("Aborting due to failure in generating detailed prompt.")
        exit(1)
    logging.info(f"Generated prompt: {detailed_prompt}")
    image_path_or_url = generate_ai_image(detailed_prompt)
    if not image_path_or_url:
        logging.error("Aborting due to failure in generating AI image.")
        exit(1)
    logging.info(f"Generated image path or URL: {image_path_or_url}")
    image_path = download_image(image_path_or_url)
    if not image_path:
        logging.error("Aborting due to failure in downloading or copying image.")
        exit(1)
    logging.info(f"Image saved to: {image_path}")
    caption = f"Generated by AI: {detailed_prompt} #AIArt #AIGenerated"
    post_to_instagram(image_path, caption)
    logging.info("Process completed successfully.")