import base64
import io
import os
import requests
from PIL import Image

def load_image_from_url(url):
    """
    Load an image from a URL.
    
    Args:
        url (str): The URL of the image.
        
    Returns:
        PIL.Image.Image: The loaded image.
    """
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content))
    else:
        raise Exception(f"Failed to load image from URL: {url}. Status code: {response.status_code}")

def load_image_from_path(path):
    """
    Load an image from a file path.
    
    Args:
        path (str): The path to the image file.
        
    Returns:
        PIL.Image.Image: The loaded image.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    return Image.open(path)

def image_to_base64(image, format="JPEG"):
    """
    Convert a PIL Image to a base64-encoded string.
    
    Args:
        image (PIL.Image.Image): The image to convert.
        format (str, optional): The image format (JPEG, PNG, etc.).
        
    Returns:
        str: The base64-encoded image string.
    """
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{img_str}"

def base64_to_image(base64_str):
    """
    Convert a base64-encoded string to a PIL Image.
    
    Args:
        base64_str (str): The base64-encoded image string.
        
    Returns:
        PIL.Image.Image: The decoded image.
    """
    # Remove the data URL prefix if present
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]
    
    img_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(img_data))

def save_image(image, path, format=None):
    """
    Save a PIL Image to a file.
    
    Args:
        image (PIL.Image.Image): The image to save.
        path (str): The path where to save the image.
        format (str, optional): The image format. If None, it will be inferred from the file extension.
        
    Returns:
        str: The path where the image was saved.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    image.save(path, format=format)
    return path

def save_video(video_data, path):
    """
    Save video data to a file.
    
    Args:
        video_data (bytes): The video data to save.
        path (str): The path where to save the video.
        
    Returns:
        str: The path where the video was saved.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    with open(path, 'wb') as f:
        f.write(video_data)
    return path

def resize_image(image, width=None, height=None, maintain_aspect=True):
    """
    Resize an image to the specified dimensions.
    
    Args:
        image (PIL.Image.Image): The image to resize.
        width (int, optional): The target width.
        height (int, optional): The target height.
        maintain_aspect (bool, optional): Whether to maintain the aspect ratio.
        
    Returns:
        PIL.Image.Image: The resized image.
    """
    if width is None and height is None:
        return image
    
    if maintain_aspect:
        if width is None:
            # Calculate width based on height while maintaining aspect ratio
            aspect_ratio = image.width / image.height
            width = int(height * aspect_ratio)
        elif height is None:
            # Calculate height based on width while maintaining aspect ratio
            aspect_ratio = image.width / image.height
            height = int(width / aspect_ratio)
        else:
            # Resize to fit within the specified dimensions while maintaining aspect ratio
            img_aspect = image.width / image.height
            target_aspect = width / height
            
            if img_aspect > target_aspect:
                # Image is wider than target, adjust height
                new_width = width
                new_height = int(width / img_aspect)
            else:
                # Image is taller than target, adjust width
                new_height = height
                new_width = int(height * img_aspect)
            
            return image.resize((new_width, new_height), Image.LANCZOS)
    
    return image.resize((width, height), Image.LANCZOS)