import requests
import os
import io
from PIL import Image
import base64

class SegmindAPI:
    """
    A Python client for the Segmind API.
    This client provides access to various AI models offered by Segmind.
    
    Args:
        api_key (str): Your Segmind API key. If not provided, it will look for SEGMIND_API_KEY environment variable.
    """
    
    BASE_URL = "https://api.segmind.com/v1/"
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("SEGMIND_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided either as an argument or as SEGMIND_API_KEY environment variable.")
        self.headers = {
            'x-api-key': self.api_key,
            'Content-Type': 'application/json'
        }
    
    def _handle_response(self, response):
        """
        Handle API response and check for errors.
        
        Args:
            response (requests.Response): The response object from the API request.
            
        Returns:
            The response content or raises an exception if there's an error.
        """
        if response.status_code == 200:
            # Check if the response is an image
            if response.headers.get('Content-Type', '').startswith('image/'):
                return Image.open(io.BytesIO(response.content))
            # Check if the response is JSON
            try:
                return response.json()
            except ValueError:
                return response.content
        else:
            error_message = f"API request failed with status code {response.status_code}"
            try:
                error_data = response.json()
                if 'error' in error_data:
                    error_message += f": {error_data['error']}"
            except ValueError:
                pass
            raise Exception(error_message)
    
    def get_remaining_credits(self, response):
        """
        Get the remaining credits from the response headers.
        
        Args:
            response (requests.Response): The response object from the API request.
            
        Returns:
            int: The number of remaining credits or None if not available.
        """
        return response.headers.get('x-remaining-credits')
    
    def text_to_image(self, model_name, **params):
        """
        Generate an image from a text prompt using the specified model.
        
        Args:
            model_name (str): The name of the model to use.
            **params: Additional parameters for the model.
            
        Returns:
            PIL.Image.Image: The generated image.
        """
        url = f"{self.BASE_URL}{model_name}"
        response = requests.post(url, json=params, headers=self.headers)
        return self._handle_response(response)
    
    def image_to_image(self, model_name, image_url=None, image_path=None, image_base64=None, **params):
        """
        Generate an image from another image using the specified model.
        
        Args:
            model_name (str): The name of the model to use.
            image_url (str, optional): URL of the input image.
            image_path (str, optional): Path to the input image file.
            image_base64 (str, optional): Base64-encoded image data.
            **params: Additional parameters for the model.
            
        Returns:
            PIL.Image.Image: The generated image.
        """
        if sum(x is not None for x in [image_url, image_path, image_base64]) != 1:
            raise ValueError("Exactly one of image_url, image_path, or image_base64 must be provided.")
        
        url = f"{self.BASE_URL}{model_name}"
        
        if image_url:
            params['image'] = image_url
            response = requests.post(url, json=params, headers=self.headers)
        elif image_path:
            with open(image_path, 'rb') as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
            params['image'] = f"data:image/jpeg;base64,{img_data}"
            response = requests.post(url, json=params, headers=self.headers)
        elif image_base64:
            params['image'] = image_base64
            response = requests.post(url, json=params, headers=self.headers)
        
        return self._handle_response(response)
    
    # Specific model implementations
    
    def sdxl(self, prompt, negative_prompt=None, steps=None, seed=None, aspect_ratio=None, base64=False, **kwargs):
        """
        Generate an image using Stable Diffusion XL 1.0 model.
        
        Args:
            prompt (str): The text prompt for image generation.
            negative_prompt (str, optional): Text to exclude from the generated image.
            steps (int, optional): Number of diffusion steps (higher = better quality but slower).
            seed (int, optional): Random seed for reproducibility.
            aspect_ratio (str, optional): Aspect ratio of the output image (e.g., "1:1", "16:9").
            base64 (bool, optional): Whether to return the image as base64 string.
            **kwargs: Additional parameters for the model.
            
        Returns:
            PIL.Image.Image: The generated image.
        """
        params = {
            'prompt': prompt,
            'base64': base64,
            **{k: v for k, v in {
                'negative_prompt': negative_prompt,
                'steps': steps,
                'seed': seed,
                'aspect_ratio': aspect_ratio,
            }.items() if v is not None},
            **kwargs
        }
        return self.text_to_image('sdxl1.0-txt2img', **params)
    
    def sd_outpainting(self, image_url=None, image_path=None, image_base64=None, prompt=None, **kwargs):
        """
        Extend an image beyond its original boundaries using Stable Diffusion Outpainting.
        
        Args:
            image_url (str, optional): URL of the input image.
            image_path (str, optional): Path to the input image file.
            image_base64 (str, optional): Base64-encoded image data.
            prompt (str, optional): Text prompt to guide the outpainting.
            **kwargs: Additional parameters for the model.
            
        Returns:
            PIL.Image.Image: The outpainted image.
        """
        params = {}
        if prompt:
            params['prompt'] = prompt
        params.update(kwargs)
        
        return self.image_to_image('sd-outpainting', 
                                  image_url=image_url, 
                                  image_path=image_path, 
                                  image_base64=image_base64, 
                                  **params)
    
    def qr_generator(self, prompt, qr_text, **kwargs):
        """
        Generate a stylized QR code based on a text prompt.
        
        Args:
            prompt (str): The text prompt for styling the QR code.
            qr_text (str): The text/URL to encode in the QR code.
            **kwargs: Additional parameters for the model.
            
        Returns:
            PIL.Image.Image: The generated QR code image.
        """
        params = {
            'prompt': prompt,
            'qr_text': qr_text,
            **kwargs
        }
        return self.text_to_image('qr-code-generator', **params)
    
    def word2img(self, image_url=None, image_path=None, image_base64=None, prompt=None, **kwargs):
        """
        Transform an image based on a text prompt.
        
        Args:
            image_url (str, optional): URL of the input image.
            image_path (str, optional): Path to the input image file.
            image_base64 (str, optional): Base64-encoded image data.
            prompt (str, optional): Text prompt to guide the transformation.
            **kwargs: Additional parameters for the model.
            
        Returns:
            PIL.Image.Image: The transformed image.
        """
        params = {}
        if prompt:
            params['prompt'] = prompt
        params.update(kwargs)
        
        return self.image_to_image('word2img', 
                                  image_url=image_url, 
                                  image_path=image_path, 
                                  image_base64=image_base64, 
                                  **params)
    
    def background_removal(self, image_url=None, image_path=None, image_base64=None, **kwargs):
        """
        Remove the background from an image.
        
        Args:
            image_url (str, optional): URL of the input image.
            image_path (str, optional): Path to the input image file.
            image_base64 (str, optional): Base64-encoded image data.
            **kwargs: Additional parameters for the model.
            
        Returns:
            PIL.Image.Image: The image with background removed.
        """
        return self.image_to_image('background-removal', 
                                 image_url=image_url, 
                                 image_path=image_path, 
                                 image_base64=image_base64, 
                                 **kwargs)
    
    def codeformer(self, image_url=None, image_path=None, image_base64=None, **kwargs):
        """
        Restore and enhance faces in images.
        
        Args:
            image_url (str, optional): URL of the input image.
            image_path (str, optional): Path to the input image file.
            image_base64 (str, optional): Base64-encoded image data.
            **kwargs: Additional parameters for the model.
            
        Returns:
            PIL.Image.Image: The image with enhanced faces.
        """
        return self.image_to_image('codeformer', 
                                 image_url=image_url, 
                                 image_path=image_path, 
                                 image_base64=image_base64, 
                                 **kwargs)
    
    def sam(self, image_url=None, image_path=None, image_base64=None, **kwargs):
        """
        Segment objects in an image using Segment Anything Model (SAM).
        
        Args:
            image_url (str, optional): URL of the input image.
            image_path (str, optional): Path to the input image file.
            image_base64 (str, optional): Base64-encoded image data.
            **kwargs: Additional parameters for the model.
            
        Returns:
            PIL.Image.Image: The segmented image.
        """
        return self.image_to_image('sam', 
                                 image_url=image_url, 
                                 image_path=image_path, 
                                 image_base64=image_base64, 
                                 **kwargs)
    
    def face_swap(self, image_url=None, image_path=None, image_base64=None, mask_url=None, **kwargs):
        """
        Swap faces in images.
        
        Args:
            image_url (str, optional): URL of the input image.
            image_path (str, optional): Path to the input image file.
            image_base64 (str, optional): Base64-encoded image data.
            mask_url (str): URL of the mask image containing the face to swap.
            **kwargs: Additional parameters for the model.
            
        Returns:
            PIL.Image.Image: The image with swapped faces.
        """
        params = {}
        if mask_url:
            params['mask'] = mask_url
        params.update(kwargs)
        
        return self.image_to_image('face-swap', 
                                  image_url=image_url, 
                                  image_path=image_path, 
                                  image_base64=image_base64, 
                                  **params)
    
    def controlnet(self, prompt, image_url=None, image_path=None, image_base64=None, option="canny", **kwargs):
        """
        Generate images guided by control maps using ControlNet.
        
        Args:
            prompt (str): The text prompt for image generation.
            image_url (str, optional): URL of the input control image.
            image_path (str, optional): Path to the input control image file.
            image_base64 (str, optional): Base64-encoded control image data.
            option (str, optional): ControlNet option (canny, depth, openpose, scribble, softedge).
            **kwargs: Additional parameters for the model.
            
        Returns:
            PIL.Image.Image: The generated image.
        """
        params = {
            'prompt': prompt,
            'option': option,
            **kwargs
        }
        
        return self.image_to_image('controlnet', 
                                  image_url=image_url, 
                                  image_path=image_path, 
                                  image_base64=image_base64, 
                                  **params)
    
    def veo_3(self, prompt, seed=None, **kwargs):
        """
        Generate video from text using Google's Veo 3 model.
        
        Args:
            prompt (str): The text prompt describing the video content.
            seed (int, optional): Random seed for reproducibility.
            **kwargs: Additional parameters for the model.
            
        Returns:
            bytes: The generated video data.
        """
        params = {
            'prompt': prompt,
            **{k: v for k, v in {'seed': seed}.items() if v is not None},
            **kwargs
        }
        
        url = f"{self.BASE_URL}veo-3"
        response = requests.post(url, json=params, headers=self.headers)
        
        # For video, we return the raw content instead of trying to parse it
        if response.status_code == 200:
            return response.content
        else:
            error_message = f"API request failed with status code {response.status_code}"
            try:
                error_data = response.json()
                if 'error' in error_data:
                    error_message += f": {error_data['error']}"
            except ValueError:
                pass
            raise Exception(error_message)
    
    def flux_kontext_pro(self, prompt, input_image=None, seed=1, aspect_ratio="match_input_image", **kwargs):
        """
        Transform images based on text prompts using FLUX.1 Kontext Pro.
        
        Args:
            prompt (str): The text prompt describing the desired transformation.
            input_image (str, optional): URL of the input image.
            seed (int, optional): Random seed for reproducibility.
            aspect_ratio (str, optional): Aspect ratio of the output image.
            **kwargs: Additional parameters for the model.
            
        Returns:
            PIL.Image.Image: The transformed image.
        """
        params = {
            'prompt': prompt,
            'seed': seed,
            'aspect_ratio': aspect_ratio,
            **kwargs
        }
        
        if input_image:
            params['input_image'] = input_image
        
        url = f"{self.BASE_URL}flux-kontext-pro"
        response = requests.post(url, json=params, headers=self.headers)
        return self._handle_response(response)
    
    def llava_13b(self, messages):
        """
        Generate responses using the LLaVA 13B vision-language model.
        
        Args:
            messages (list): List of message objects with 'role' and 'content' keys.
                Example: [
                    {"role": "user", "content": "tell me a joke on cats"},
                    {"role": "assistant", "content": "here is a joke about cats..."},
                    {"role": "user", "content": "now a joke on dogs"}
                ]
            
        Returns:
            dict: The model's response.
        """
        params = {
            'messages': messages
        }
        
        url = f"{self.BASE_URL}llava-13b"
        response = requests.post(url, json=params, headers=self.headers)
        return self._handle_response(response)