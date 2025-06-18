import os
import time
import random
# Import directly from the local modules
from segmind_models import (
    SDXL,
    SDOutpainting,
    QRGenerator,
    Word2Img,
    BackgroundRemoval,
    Codeformer,
    SAM,
    ControlNet,
    Veo3,
    FluxKontextPro,
    LLaVA13B
)

# Set your API key
API_KEY = ""  # Replace with your actual API key if not using environment variable

# Create output directory if it doesn't exist
os.makedirs("output", exist_ok=True)


def example_sdxl():
    """Example of using SDXL for text-to-image generation."""
    print("\n=== Running SDXL Example ===")
    model = SDXL(API_KEY)
    image = model.generate(
        prompt="A beautiful landscape with mountains, lakes, and a colorful sunset, photorealistic, 4k",
        negative_prompt="blurry, low quality, distorted",
        steps=30,
        seed=42,
        aspect_ratio="16:9",
        save_path="output/sdxl_landscape.jpg"
    )
    print(f"Image generated and saved to 'output/sdxl_landscape.jpg'")
    return image


def example_qr_generator():
    """Example of using QRGenerator for stylized QR codes."""
    print("\n=== Running QR Generator Example ===")
    model = QRGenerator(API_KEY)
    image = model.generate(
        prompt="A colorful abstract painting with geometric shapes",
        qr_text="https://www.segmind.com",
        save_path="output/stylized_qr.png"
    )
    print(f"QR code generated and saved to 'output/stylized_qr.png'")
    return image


def example_background_removal():
    """Example of using BackgroundRemoval to remove image backgrounds."""
    print("\n=== Running Background Removal Example ===")
    # This example uses a sample image URL. Replace with your own image URL or path.
    image_url = "https://images.unsplash.com/photo-1494790108377-be9c29b29330"
    
    model = BackgroundRemoval(API_KEY)
    image = model.generate(
        image_url=image_url,
        save_path="output/no_background.png"
    )
    print(f"Background removed and image saved to 'output/no_background.png'")
    return image


def example_llava():
    """Example of using LLaVA13B for vision-language tasks."""
    print("\n=== Running LLaVA13B Example ===")
    model = LLaVA13B(API_KEY)
    response = model.generate([
        {"role": "user", "content": "tell me a joke about programming"}
    ])
    print("LLaVA13B Response:")
    print(response)
    return response


def example_veo3():
    """Example of using Veo3 for text-to-video generation."""
    print("\n=== Running Veo3 Example ===")
    model = Veo3(API_KEY)
    video_data = model.generate(
        prompt="A timelapse of a blooming flower in a garden",
        save_path="output/flower_timelapse.mp4"
    )
    print(f"Video generated and saved to 'output/flower_timelapse.mp4'")
    return video_data


def run_with_backoff(func, max_retries=5, initial_delay=1):
    """
    Run a function with exponential backoff for handling rate limit errors.
    
    Args:
        func: The function to run
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before retrying
        
    Returns:
        The result of the function if successful
    """
    retries = 0
    delay = initial_delay
    
    while retries <= max_retries:
        try:
            return func()
        except Exception as e:
            error_message = str(e)
            if "429" in error_message and retries < max_retries:
                # Rate limit error, implement backoff
                jitter = random.uniform(0, 0.1) * delay  # Add some randomness to the delay
                wait_time = delay + jitter
                print(f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds... (Attempt {retries+1}/{max_retries})")
                time.sleep(wait_time)
                delay *= 2  # Exponential backoff
                retries += 1
            else:
                # Other error or max retries reached
                raise e
    
    raise Exception(f"Failed after {max_retries} retries")


def run_examples():
    """Run all examples."""
    print("Starting Segmind API Examples...\n")
    print("Note: These examples require a valid Segmind API key with sufficient credits.")
    print("If you encounter rate limit errors (429), you may need to:")
    print("  1. Wait a while before trying again")
    print("  2. Check your API key's remaining credits")
    print("  3. Upgrade your API plan if you need higher rate limits\n")
    
    try:
        # Run examples with backoff
        run_with_backoff(example_sdxl)
        # Uncomment to run other examples
        # run_with_backoff(example_qr_generator)
        # run_with_backoff(example_background_removal)
        # run_with_backoff(example_llava_13b)
        # run_with_backoff(example_veo_3)
        
        print("\nAll examples completed successfully!")
    except Exception as e:
        if "429" in str(e):
            print("\nError: Rate limit exceeded (429 Too Many Requests)")
            print("The Segmind API is currently rate limiting your requests.")
            print("Please wait a while before trying again or check your API key's quota.")
        else:
            print(f"\nError running examples: {e}")


if __name__ == "__main__":
    run_examples()