import os
import unittest
from unittest.mock import patch, MagicMock
from PIL import Image
import io

# Import the modules to test
from segmind_api import SegmindAPI
from segmind_models import SDXL, BackgroundRemoval, QRGenerator


class TestSegmindAPI(unittest.TestCase):
    """Test cases for the SegmindAPI class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.api_key = ""  # Test API key
        self.api = SegmindAPI(self.api_key)
    
    def test_init_with_api_key(self):
        """Test initialization with API key."""
        api = SegmindAPI(self.api_key)
        self.assertEqual(api.api_key, self.api_key)
        self.assertEqual(api.headers['x-api-key'], self.api_key)
    
    @patch.dict(os.environ, {"SEGMIND_API_KEY": "test_env_key"})
    def test_init_with_env_var(self):
        """Test initialization with environment variable."""
        api = SegmindAPI()
        self.assertEqual(api.api_key, "test_env_key")
    
    def test_init_without_api_key(self):
        """Test initialization without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                SegmindAPI()
    
    @patch('requests.post')
    def test_text_to_image(self, mock_post):
        """Test text_to_image method."""
        # Create a mock response with an image
        mock_image = Image.new('RGB', (100, 100), color='red')
        img_byte_arr = io.BytesIO()
        mock_image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {'Content-Type': 'image/jpeg'}
        mock_response.content = img_byte_arr
        mock_post.return_value = mock_response
        
        # Call the method
        result = self.api.text_to_image('sdxl1.0', prompt="test prompt")
        
        # Verify the result
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, (100, 100))
        
        # Verify the API call
        mock_post.assert_called_once_with(
            "https://api.segmind.com/v1/sdxl1.0",
            json={"prompt": "test prompt"},
            headers=self.api.headers
        )
    
    @patch('requests.post')
    def test_image_to_image(self, mock_post):
        """Test image_to_image method with image URL."""
        # Create a mock response with an image
        mock_image = Image.new('RGB', (100, 100), color='blue')
        img_byte_arr = io.BytesIO()
        mock_image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {'Content-Type': 'image/jpeg'}
        mock_response.content = img_byte_arr
        mock_post.return_value = mock_response
        
        # Call the method
        result = self.api.image_to_image(
            'background-removal',
            image_url="https://example.com/image.jpg"
        )
        
        # Verify the result
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, (100, 100))
        
        # Verify the API call
        mock_post.assert_called_once_with(
            "https://api.segmind.com/v1/background-removal",
            json={"image": "https://example.com/image.jpg"},
            headers=self.api.headers
        )
    
    @patch('requests.post')
    def test_error_handling(self, mock_post):
        """Test error handling."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"error": "Invalid API key"}
        mock_post.return_value = mock_response
        
        # Call the method and expect an exception
        with self.assertRaises(Exception) as context:
            self.api.text_to_image('sdxl1.0', prompt="test prompt")
        
        # Verify the exception message
        self.assertIn("API request failed with status code 401: Invalid API key", str(context.exception))


class TestModelClasses(unittest.TestCase):
    """Test cases for the model classes."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.api_key = "SG_14ad6b22f5e1342e"  # Test API key
    
    @patch('segmind_api.SegmindAPI.sdxl')
    def test_sdxl_model(self, mock_sdxl):
        """Test SDXL model class."""
        # Create a mock image
        mock_image = Image.new('RGB', (100, 100), color='green')
        mock_sdxl.return_value = mock_image
        
        # Create the model and call generate
        model = SDXL(self.api_key)
        result = model.generate(
            prompt="test prompt",
            negative_prompt="bad quality",
            steps=30,
            seed=42
        )
        
        # Verify the result
        self.assertEqual(result, mock_image)
        
        # Verify the API call
        mock_sdxl.assert_called_once_with(
            prompt="test prompt",
            negative_prompt="bad quality",
            steps=30,
            seed=42
        )
    
    @patch('segmind_api.SegmindAPI.background_removal')
    def test_background_removal_model(self, mock_bg_removal):
        """Test BackgroundRemoval model class."""
        # Create a mock image
        mock_image = Image.new('RGB', (100, 100), color='white')
        mock_bg_removal.return_value = mock_image
        
        # Create the model and call generate
        model = BackgroundRemoval(self.api_key)
        result = model.generate(image_url="https://example.com/image.jpg")
        
        # Verify the result
        self.assertEqual(result, mock_image)
        
        # Verify the API call
        mock_bg_removal.assert_called_once_with(
            image_url="https://example.com/image.jpg"
        )
    
    @patch('segmind_api.SegmindAPI.qr_generator')
    def test_qr_generator_model(self, mock_qr_generator):
        """Test QRGenerator model class."""
        # Create a mock image
        mock_image = Image.new('RGB', (100, 100), color='black')
        mock_qr_generator.return_value = mock_image
        
        # Create the model and call generate
        model = QRGenerator(self.api_key)
        result = model.generate(
            prompt="colorful QR",
            qr_text="https://example.com"
        )
        
        # Verify the result
        self.assertEqual(result, mock_image)
        
        # Verify the API call
        mock_qr_generator.assert_called_once_with(
            prompt="colorful QR",
            qr_text="https://example.com"
        )


if __name__ == '__main__':
    unittest.main()
