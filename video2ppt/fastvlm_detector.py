"""
FastVLM-based PPT content detector for video frames.

This module provides functionality to detect PPT slides in video frames using the FastVLM model.
"""

import os
import logging
import time
from typing import Optional, Tuple, Dict, Any, List, Union

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, AutoImageProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FastVLMDetector:
    """Detect PPT content in images using FastVLM."""
    
    # Keywords that indicate PPT content
    PPT_KEYWORDS = [
        'slide', 'presentation', 'powerpoint', 'ppt', 'title', 'bullet',
        'heading', 'subheading', 'content', 'text', 'diagram', 'chart',
        'graph', 'table', 'list', 'agenda', 'outline', 'summary',
        'layout', 'template', 'design', 'background', 'theme', 'font',
        'color scheme', 'animation', 'transition', 'speaker notes',
        'master slide', 'placeholder', 'smartart', 'organization chart',
        'process', 'hierarchy', 'relationship', 'matrix', 'pyramid'
    ]
    
    # Negative keywords that might indicate non-PPT content
    NEGATIVE_KEYWORDS = [
        'photo', 'photograph', 'screenshot', 'desktop', 'window', 'browser',
        'application', 'video', 'player', 'media', 'game', 'menu', 'toolbar',
        'icon', 'notification', 'taskbar', 'dock', 'finder', 'explorer',
        'file manager', 'terminal', 'console', 'code', 'editor', 'ide'
    ]
    
    def __init__(self, model_path: str, device: Optional[str] = None, debug: bool = False):
        """Initialize the detector.
        
        Args:
            model_path: Path to the model directory
            device: Device to run the model on ('cuda', 'mps', or 'cpu')
            debug: Enable debug mode for more verbose logging
        """
        self.model_path = model_path
        self.device = device or self._get_available_device()
        self.debug = debug
        
        # Initialize model components
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.processor = None  # For BLIP model
        self.model_type = None  # Will be set to 'fastvlm' or 'blip'
        
        # Performance tracking
        self.last_inference_time = 0
        self.total_inferences = 0
        
        # Load model and processor
        self._load_model()
        
        # Warm up the model with a dummy inference
        if self.model is not None:
            self._warm_up()
    
    def _get_available_device(self) -> str:
        """Get the best available device for inference."""
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'  # For Apple Silicon
        return 'cpu'
    
    def _load_model(self):
        """Load the model for image analysis.
        
        This method attempts to load the FastVLM model if available, but falls back
        to using a standard vision model if FastVLM is not compatible.
        """
        try:
            logger.info(f"Loading model from {self.model_path}...")
            start_time = time.time()
            
            # First try to load with trust_remote_code=True for FastVLM
            try:
                # Load model with appropriate settings
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16 if 'cuda' in self.device else torch.float32,
                    low_cpu_mem_usage=True,
                    device_map="auto",
                    trust_remote_code=True
                )
                
                # Load tokenizer and image processor separately for better compatibility
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                )
                
                self.image_processor = AutoImageProcessor.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                )
                
                # Set model to eval mode
                self.model.eval()
                
                logger.info(f"Successfully loaded FastVLM model in {time.time() - start_time:.2f}s")
                self.model_type = "fastvlm"
                
            except (ValueError, KeyError) as e:
                logger.warning(f"Failed to load FastVLM model: {e}. Falling back to standard vision model.")
                
                # Fall back to a standard vision-language model
                from transformers import BlipProcessor, BlipForConditionalGeneration
                
                self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                
                # Move model to appropriate device
                self.model = self.model.to(self.device)
                self.model.eval()
                
                logger.info(f"Successfully loaded BLIP model in {time.time() - start_time:.2f}s")
                self.model_type = "blip"
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded on {self.device} in {load_time:.2f} seconds")
            
            # Warm up the model with a dummy inference
            self._warmup()
            
        except Exception as e:
            logger.error(f"Failed to load FastVLM model: {e}")
            raise
    
    def _warm_up(self):
        """Warm up the model with a dummy inference to ensure it's ready for real-time use."""
        try:
            logger.info("Warming up model with dummy inference...")
            # Create a dummy image for inference
            dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
            # Run a quick inference (result is discarded)
            _, _, _ = self.is_ppt_content(dummy_image, confidence_threshold=0.5)
            logger.info("Model warm-up completed successfully")
        except Exception as e:
            logger.warning(f"Model warm-up failed: {e}")
            if self.debug:
                import traceback
                logger.warning(traceback.format_exc())
    
    def _generate_description(self, image: np.ndarray, prompt: str) -> str:
        """Generate a description for an image using the model.
        
        Args:
            image: Input image as a numpy array (BGR format)
            prompt: Text prompt to guide the image description
            
        Returns:
            Generated description text
        """
        try:
            # Convert BGR to RGB and to PIL Image
            image_rgb = image[..., ::-1]  # BGR to RGB
            pil_image = Image.fromarray(image_rgb)
            
            # Process the image based on model type
            start_time = time.time()
            
            if hasattr(self, 'model_type') and self.model_type == 'blip':
                # Using BLIP model
                inputs = self.processor(pil_image, prompt, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_new_tokens=100)
                    
                description = self.processor.decode(outputs[0], skip_special_tokens=True)
                self.total_inferences += 1
                
                # Decode the output
                description = self.tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True
                )
                
                # Remove the input prompt from the output
                description = description.replace(prompt, '').strip()
                
                return description
                
        except Exception as e:
            logger.error(f"Error generating description: {e}")
            if hasattr(e, 'message'):
                return f"Error: {e.message}"
            return f"Error: {str(e)}"
    
    def is_ppt_content(self, image: np.ndarray, confidence_threshold: float = 0.3) -> Tuple[bool, float, str]:
        """Check if the image contains PPT content.
        
        Args:
            image: Input image as a numpy array (BGR format from OpenCV)
            confidence_threshold: Minimum confidence score to consider as PPT content
            
        Returns:
            Tuple of (is_ppt, confidence, description)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        try:
            # Convert BGR to RGB and to PIL Image
            image_rgb = image[..., ::-1]  # BGR to RGB
            pil_image = Image.fromarray(image_rgb)
            
            # Generate description based on model type
            if hasattr(self, 'model_type') and self.model_type == 'blip':
                # Using BLIP model
                prompt = "Is this a presentation slide or PowerPoint? Describe in detail."
                inputs = self.processor(pil_image, prompt, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_new_tokens=100)
                    
                description = self.processor.decode(outputs[0], skip_special_tokens=True)
                
            else:
                # Using FastVLM model
                prompt = "Describe this image in detail, focusing on whether it contains presentation slides or similar content."
                
                # Process the image based on available processors
                if hasattr(self, 'processor') and self.processor is not None:
                    inputs = self.processor(images=pil_image, text=prompt, return_tensors="pt").to(self.device)
                elif hasattr(self, 'tokenizer') and hasattr(self, 'image_processor'):
                    # Separate tokenizer and image processor
                    pixel_values = self.image_processor(images=pil_image, return_tensors="pt").pixel_values.to(self.device)
                    input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
                    inputs = {"pixel_values": pixel_values, "input_ids": input_ids}
                else:
                    raise RuntimeError("No valid processor found for the model")
                
                # Generate description
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                    )
                
                # Decode the output
                if hasattr(self, 'processor') and self.processor is not None:
                    description = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
                else:
                    description = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Check for PPT-related keywords
            description_lower = description.lower()
            keyword_matches = sum(keyword in description_lower for keyword in self.PPT_KEYWORDS)
            
            # Check for negative keywords
            negative_matches = sum(keyword in description_lower for keyword in self.NEGATIVE_KEYWORDS)
            
            # Calculate confidence score with penalty for negative keywords
            confidence = min(1.0, (keyword_matches - (negative_matches * 0.5)) / 3.0)  # Cap at 1.0
            confidence = max(0.0, confidence)  # Ensure non-negative
            
            # Additional heuristics for PPT detection
            if 'slide' in description_lower and 'presentation' in description_lower:
                confidence += 0.2
            if 'bullet point' in description_lower or 'bullet points' in description_lower:
                confidence += 0.1
            if 'title' in description_lower and 'subtitle' in description_lower:
                confidence += 0.1
                
            confidence = min(1.0, confidence)  # Cap at 1.0 again after adjustments
            is_ppt = confidence >= confidence_threshold
            
            # Log the detection result
            logger.debug(f"PPT detection: confidence={confidence:.2f}, is_ppt={is_ppt}")
            
            return is_ppt, confidence, description
            
        except Exception as e:
            logger.error(f"Error during PPT detection: {e}")
            if self.debug:
                import traceback
                logger.error(traceback.format_exc())
            return False, 0.0, f"Error: {str(e)}"
    
    def process_frame(self, frame: np.ndarray, confidence_threshold: float = 0.3) -> Dict[str, Any]:
        """Process a single frame and return detection results.
        
        Args:
            frame: Input frame as a numpy array (BGR format)
            confidence_threshold: Minimum confidence to consider as PPT content
            
        Returns:
            Dictionary containing detection results
        """
        is_ppt, confidence, description = self.is_ppt_content(frame, confidence_threshold)
        
        return {
            'is_ppt': is_ppt,
            'confidence': confidence,
            'description': description,
            'timestamp': None  # Can be set by the caller
        }


def load_fastvlm_model(model_path: str, device: Optional[str] = None) -> FastVLMDetector:
    """Load the FastVLM model and return a detector instance.
    
    Args:
        model_path: Path to the FastVLM model directory
        device: Device to run the model on ('cuda', 'mps', or 'cpu')
        
    Returns:
        Initialized FastVLMDetector instance
    """
    return FastVLMDetector(model_path=model_path, device=device)


# Example usage
if __name__ == "__main__":
    import cv2
    
    # Example usage
    model_path = "./checkpoints/fastvlm_1.5b_stage3"
    detector = load_fastvlm_model(model_path)
    
    # Load an example image
    image_path = "example.png"
    if os.path.exists(image_path):
        image = cv2.imread(image_path)
        if image is not None:
            result = detector.process_frame(image)
            print(f"Is PPT: {result['is_ppt']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Description: {result['description']}")
        else:
            print(f"Failed to load image: {image_path}")
    else:
        print(f"Image not found: {image_path}")
