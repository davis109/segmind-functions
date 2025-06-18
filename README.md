# Sugmind: Python Client for Segmind APIs

A comprehensive Python client for interacting with Segmind's AI models and APIs. This package provides easy access to various generative AI models offered by Segmind, including text-to-image, image-to-image, background removal, face enhancement, and more.

## Installation

```bash
pip install -e .
```

Or install directly from the requirements:

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from sugmind import SDXL

# Initialize with your API key
api_key = "SG_14ad6b22f5e1342e"  # Replace with your actual API key
model = SDXL(api_key)

# Generate an image from a text prompt
image = model.generate(
    prompt="A beautiful sunset over the mountains, photorealistic, 4k",
    save_path="sunset.jpg"  # Optional: save the image to a file
)

# Display the image
image.show()
```

## Available Models

The package provides access to the following Segmind models:

### Text-to-Image Models

- **SDXL**: Stable Diffusion XL 1.0 for high-quality image generation
- **SD2_1**: Alias for SDXL (for backward compatibility)

### Image-to-Image Models

- **SDOutpainting**: Extend images beyond their original boundaries
- **Word2Img**: Transform images based on text prompts
- **ControlNet**: Generate images guided by control maps (canny, depth, openpose, etc.)
- **FluxKontextPro**: Advanced image transformation based on text prompts
- **Kadinsky**: Alias for FluxKontextPro (for backward compatibility)

### Image Enhancement Models

- **BackgroundRemoval**: Remove backgrounds from images
- **Codeformer**: Restore and enhance faces in images
- **FaceSwap**: Swap faces in images

### Segmentation Models

- **SAM**: Segment Anything Model for object segmentation in images

### Video Generation Models

- **Veo3**: Generate videos from text descriptions

### Multimodal Models

- **LLaVA13B**: Vision-language model for generating responses to text and image inputs

### Utility Models

- **QRGenerator**: Generate stylized QR codes based on text prompts

## Usage Examples

### Generate an Image with SDXL

```python
from sugmind import SDXL

model = SDXL(api_key)
image = model.generate(
    prompt="A cyberpunk cityscape at night with neon lights",
    negative_prompt="blurry, low quality",
    steps=30,
    seed=42,
    aspect_ratio="16:9"
)
image.save("cityscape.png")
```

### Remove Background from an Image

```python
from sugmind import BackgroundRemoval

model = BackgroundRemoval(api_key)
image = model.generate(
    image_path="portrait.jpg",
    save_path="portrait_no_bg.png"
)
```

### Generate a Stylized QR Code

```python
from sugmind import QRGenerator

model = QRGenerator(api_key)
image = model.generate(
    prompt="A colorful abstract painting with geometric shapes",
    qr_text="https://example.com",
    save_path="stylized_qr.png"
)
```

### Generate a Video with Veo3

```python
from sugmind import Veo3

model = Veo3(api_key)
video_data = model.generate(
    prompt="A timelapse of a blooming flower in a garden",
    save_path="flower_timelapse.mp4"
)
```

### Use LLaVA 13B for Vision-Language Tasks

```python
from sugmind import LLaVA13B

model = LLaVA13B(api_key)
response = model.generate([
    {"role": "user", "content": "What can you see in this image?"},
    # Add more messages as needed
])
print(response)
```

## Advanced Usage

### Using the Base API Client

For more advanced use cases, you can use the base `SegmindAPI` class directly:

```python
from sugmind import SegmindAPI

api = SegmindAPI(api_key)

# Call any Segmind API endpoint
response = api.text_to_image(
    model_name="custom-model-endpoint",
    prompt="Your prompt here",
    # Additional parameters
)
```

### Working with Local Images

```python
from sugmind import Word2Img
from sugmind_utils import load_image_from_path, image_to_base64

model = Word2Img(api_key)

# Load a local image
image = load_image_from_path("input.jpg")

# Convert to base64 if needed
image_base64 = image_to_base64(image)

# Use the model
result = model.generate(
    prompt="Transform this into a watercolor painting",
    image_path="input.jpg"
)
result.save("output.jpg")
```
## Error Handling

```python
try:
    image = model.generate(prompt="Your prompt here")
except Exception as e:
    print(f"Error: {e}")
```

### Handling Rate Limits

The Segmind API implements rate limiting to ensure fair usage. If you encounter a `429 Too Many Requests` error, it means you've exceeded the allowed number of requests in a given time period. Here are some strategies to handle rate limits:

1. **Implement exponential backoff**: When you receive a 429 error, wait for a short period before retrying, and increase the wait time exponentially with each retry.

2. **Check your API quota**: Monitor your remaining credits by checking the `x-remaining-credits` header in API responses.

3. **Optimize your requests**: Batch operations when possible and avoid making unnecessary API calls.

4. **Upgrade your plan**: If you consistently hit rate limits, consider upgrading to a higher tier plan with increased limits.

Example implementation of exponential backoff:

```python
import time
import random

def call_with_backoff(func, max_retries=5, initial_delay=1):
    retries = 0
    delay = initial_delay
    
    while retries <= max_retries:
        try:
            return func()
        except Exception as e:
            if "429" in str(e) and retries < max_retries:
                # Add jitter to avoid thundering herd problem
                wait_time = delay + random.uniform(0, 0.1) * delay
                print(f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
                delay *= 2  # Exponential backoff
                retries += 1
            else:
                raise e
    
    raise Exception(f"Failed after {max_retries} retries")
```

## Credits

This package is a client for the Segmind API. For more information about Segmind and their services, visit [https://www.segmind.com/](https://www.segmind.com/).