from segmind_api import SegmindAPI
from segmind_utils import load_image_from_path, load_image_from_url, image_to_base64, save_image, save_video

class ModelBase:
    """
    Base class for all Segmind models.
    
    Args:
        api_key (str): Your Segmind API key.
    """
    
    def __init__(self, api_key=None):
        self.api = SegmindAPI(api_key)


class SDXL(ModelBase):
    """
    Stable Diffusion XL 1.0 model for text-to-image generation.
    
    Args:
        api_key (str): Your Segmind API key.
    """
    
    def generate(self, prompt, negative_prompt=None, steps=None, seed=None, aspect_ratio=None, save_path=None, **kwargs):
        """
        Generate an image from a text prompt.
        
        Args:
            prompt (str): The text prompt for image generation.
            negative_prompt (str, optional): Text to exclude from the generated image.
            steps (int, optional): Number of diffusion steps (higher = better quality but slower).
            seed (int, optional): Random seed for reproducibility.
            aspect_ratio (str, optional): Aspect ratio of the output image (e.g., "1:1", "16:9").
            save_path (str, optional): Path to save the generated image.
            **kwargs: Additional parameters for the model.
            
        Returns:
            PIL.Image.Image: The generated image.
        """
        params = {
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'steps': steps,
            'seed': seed,
            **kwargs
        }
        
        # Only add aspect_ratio if it's not None
        if aspect_ratio is not None:
            params['aspect_ratio'] = aspect_ratio
            
        image = self.api.sdxl(**params)
        
        if save_path:
            save_image(image, save_path)
        
        return image


class SDOutpainting(ModelBase):
    """
    Stable Diffusion Outpainting model for extending images beyond their boundaries.
    
    Args:
        api_key (str): Your Segmind API key.
    """
    
    def generate(self, prompt=None, image_url=None, image_path=None, save_path=None, **kwargs):
        """
        Extend an image beyond its original boundaries.
        
        Args:
            prompt (str, optional): Text prompt to guide the outpainting.
            image_url (str, optional): URL of the input image.
            image_path (str, optional): Path to the input image file.
            save_path (str, optional): Path to save the generated image.
            **kwargs: Additional parameters for the model.
            
        Returns:
            PIL.Image.Image: The outpainted image.
        """
        if image_path and not image_url:
            # Convert local image to base64
            image = load_image_from_path(image_path)
            image_base64 = image_to_base64(image)
            result = self.api.sd_outpainting(prompt=prompt, image_base64=image_base64, **kwargs)
        else:
            result = self.api.sd_outpainting(prompt=prompt, image_url=image_url, **kwargs)
        
        if save_path:
            save_image(result, save_path)
        
        return result


class QRGenerator(ModelBase):
    """
    QR code generator that creates stylized QR codes based on text prompts.
    
    Args:
        api_key (str): Your Segmind API key.
    """
    
    def generate(self, prompt, qr_text, save_path=None, **kwargs):
        """
        Generate a stylized QR code.
        
        Args:
            prompt (str): The text prompt for styling the QR code.
            qr_text (str): The text/URL to encode in the QR code.
            save_path (str, optional): Path to save the generated QR code image.
            **kwargs: Additional parameters for the model.
            
        Returns:
            PIL.Image.Image: The generated QR code image.
        """
        result = self.api.qr_generator(prompt=prompt, qr_text=qr_text, **kwargs)
        
        if save_path:
            save_image(result, save_path)
        
        return result


class Word2Img(ModelBase):
    """
    Word2Img model for transforming images based on text prompts.
    
    Args:
        api_key (str): Your Segmind API key.
    """
    
    def generate(self, prompt, image_url=None, image_path=None, save_path=None, **kwargs):
        """
        Transform an image based on a text prompt.
        
        Args:
            prompt (str): Text prompt to guide the transformation.
            image_url (str, optional): URL of the input image.
            image_path (str, optional): Path to the input image file.
            save_path (str, optional): Path to save the generated image.
            **kwargs: Additional parameters for the model.
            
        Returns:
            PIL.Image.Image: The transformed image.
        """
        if image_path and not image_url:
            # Convert local image to base64
            image = load_image_from_path(image_path)
            image_base64 = image_to_base64(image)
            result = self.api.word2img(prompt=prompt, image_base64=image_base64, **kwargs)
        else:
            result = self.api.word2img(prompt=prompt, image_url=image_url, **kwargs)
        
        if save_path:
            save_image(result, save_path)
        
        return result


class BackgroundRemoval(ModelBase):
    """
    Background removal model for removing backgrounds from images.
    
    Args:
        api_key (str): Your Segmind API key.
    """
    
    def generate(self, image_url=None, image_path=None, save_path=None, **kwargs):
        """
        Remove the background from an image.
        
        Args:
            image_url (str, optional): URL of the input image.
            image_path (str, optional): Path to the input image file.
            save_path (str, optional): Path to save the processed image.
            **kwargs: Additional parameters for the model.
            
        Returns:
            PIL.Image.Image: The image with background removed.
        """
        if image_path and not image_url:
            # Convert local image to base64
            image = load_image_from_path(image_path)
            image_base64 = image_to_base64(image)
            result = self.api.background_removal(image_base64=image_base64, **kwargs)
        else:
            result = self.api.background_removal(image_url=image_url, **kwargs)
        
        if save_path:
            save_image(result, save_path)
        
        return result


class Codeformer(ModelBase):
    """
    Codeformer model for restoring and enhancing faces in images.
    
    Args:
        api_key (str): Your Segmind API key.
    """
    
    def generate(self, image_url=None, image_path=None, save_path=None, **kwargs):
        """
        Restore and enhance faces in an image.
        
        Args:
            image_url (str, optional): URL of the input image.
            image_path (str, optional): Path to the input image file.
            save_path (str, optional): Path to save the processed image.
            **kwargs: Additional parameters for the model.
            
        Returns:
            PIL.Image.Image: The image with enhanced faces.
        """
        if image_path and not image_url:
            # Convert local image to base64
            image = load_image_from_path(image_path)
            image_base64 = image_to_base64(image)
            result = self.api.codeformer(image_base64=image_base64, **kwargs)
        else:
            result = self.api.codeformer(image_url=image_url, **kwargs)
        
        if save_path:
            save_image(result, save_path)
        
        return result


class SAM(ModelBase):
    """
    Segment Anything Model (SAM) for segmenting objects in images.
    
    Args:
        api_key (str): Your Segmind API key.
    """
    
    def generate(self, image_url=None, image_path=None, save_path=None, **kwargs):
        """
        Segment objects in an image.
        
        Args:
            image_url (str, optional): URL of the input image.
            image_path (str, optional): Path to the input image file.
            save_path (str, optional): Path to save the segmented image.
            **kwargs: Additional parameters for the model.
            
        Returns:
            PIL.Image.Image: The segmented image.
        """
        if image_path and not image_url:
            # Convert local image to base64
            image = load_image_from_path(image_path)
            image_base64 = image_to_base64(image)
            result = self.api.sam(image_base64=image_base64, **kwargs)
        else:
            result = self.api.sam(image_url=image_url, **kwargs)
        
        if save_path:
            save_image(result, save_path)
        
        return result


class FaceSwap(ModelBase):
    """
    FaceSwap model for swapping faces in images.
    
    Args:
        api_key (str): Your Segmind API key.
    """
    
    def generate(self, image_url=None, image_path=None, mask_url=None, save_path=None, **kwargs):
        """
        Swap faces in an image.
        
        Args:
            image_url (str, optional): URL of the input image.
            image_path (str, optional): Path to the input image file.
            mask_url (str): URL of the mask image containing the face to swap.
            save_path (str, optional): Path to save the processed image.
            **kwargs: Additional parameters for the model.
            
        Returns:
            PIL.Image.Image: The image with swapped faces.
        """
        if image_path and not image_url:
            # Convert local image to base64
            image = load_image_from_path(image_path)
            image_base64 = image_to_base64(image)
            result = self.api.face_swap(image_base64=image_base64, mask_url=mask_url, **kwargs)
        else:
            result = self.api.face_swap(image_url=image_url, mask_url=mask_url, **kwargs)
        
        if save_path:
            save_image(result, save_path)
        
        return result


class ControlNet(ModelBase):
    """
    ControlNet model for generating images guided by control maps.
    
    Args:
        api_key (str): Your Segmind API key.
    """
    
    def generate(self, prompt, image_url=None, image_path=None, option="canny", save_path=None, **kwargs):
        """
        Generate an image guided by a control map.
        
        Args:
            prompt (str): The text prompt for image generation.
            image_url (str, optional): URL of the input control image.
            image_path (str, optional): Path to the input control image file.
            option (str, optional): ControlNet option (canny, depth, openpose, scribble, softedge).
            save_path (str, optional): Path to save the generated image.
            **kwargs: Additional parameters for the model.
            
        Returns:
            PIL.Image.Image: The generated image.
        """
        if image_path and not image_url:
            # Convert local image to base64
            image = load_image_from_path(image_path)
            image_base64 = image_to_base64(image)
            result = self.api.controlnet(prompt=prompt, image_base64=image_base64, option=option, **kwargs)
        else:
            result = self.api.controlnet(prompt=prompt, image_url=image_url, option=option, **kwargs)
        
        if save_path:
            save_image(result, save_path)
        
        return result


class Veo3(ModelBase):
    """
    Google's Veo 3 model for generating videos from text.
    
    Args:
        api_key (str): Your Segmind API key.
    """
    
    def generate(self, prompt, seed=None, save_path=None, **kwargs):
        """
        Generate a video from a text prompt.
        
        Args:
            prompt (str): The text prompt describing the video content.
            seed (int, optional): Random seed for reproducibility.
            save_path (str, optional): Path to save the generated video.
            **kwargs: Additional parameters for the model.
            
        Returns:
            bytes: The generated video data if save_path is None, otherwise the path where the video was saved.
        """
        video_data = self.api.veo_3(prompt=prompt, seed=seed, **kwargs)
        
        if save_path:
            return save_video(video_data, save_path)
        
        return video_data


class FluxKontextPro(ModelBase):
    """
    FLUX.1 Kontext Pro model for transforming images based on text prompts.
    
    Args:
        api_key (str): Your Segmind API key.
    """
    
    def generate(self, prompt, input_image=None, seed=1, aspect_ratio="match_input_image", save_path=None, **kwargs):
        """
        Transform an image based on a text prompt.
        
        Args:
            prompt (str): The text prompt describing the desired transformation.
            input_image (str, optional): URL of the input image.
            seed (int, optional): Random seed for reproducibility.
            aspect_ratio (str, optional): Aspect ratio of the output image.
            save_path (str, optional): Path to save the transformed image.
            **kwargs: Additional parameters for the model.
            
        Returns:
            PIL.Image.Image: The transformed image.
        """
        result = self.api.flux_kontext_pro(
            prompt=prompt,
            input_image=input_image,
            seed=seed,
            aspect_ratio=aspect_ratio,
            **kwargs
        )
        
        if save_path:
            save_image(result, save_path)
        
        return result


class LLaVA13B(ModelBase):
    """
    LLaVA 13B vision-language model for generating responses to text and image inputs.
    
    Args:
        api_key (str): Your Segmind API key.
    """
    
    def generate(self, messages):
        """
        Generate a response using the LLaVA 13B model.
        
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
        return self.api.llava_13b(messages=messages)