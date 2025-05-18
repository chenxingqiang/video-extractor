"""
PPT content detection using FastVLM.
"""
import os
import torch
from PIL import Image
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import cv2

class PPTDetector:
    """Detect PPT content in frames using FastVLM."""
    
    def __init__(self, model_path: str, device: str = None):
        """Initialize the PPT detector.
        
        Args:
            model_path: Path to the FastVLM model checkpoint
            device: Device to run the model on ('cuda', 'mps', or 'cpu')
        """
        self.model_path = model_path
        self.device = self._get_device(device)
        self.model = None
        self.processor = None
        self._load_model()
        
        # Keywords that indicate PPT content
        self.ppt_keywords = [
            'slide', 'presentation', 'powerpoint', 'ppt', 
            'bullet point', 'title', 'heading', 'text',
            'chart', 'graph', 'diagram', 'table', 'list'
        ]
    
    def _get_device(self, device: Optional[str] = None) -> torch.device:
        """Get the appropriate device for inference."""
        if device is None:
            if torch.backends.mps.is_available():
                return torch.device('mps')
            elif torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')
        return torch.device(device)
    
    def _load_model(self):
        """Load the FastVLM model and processor."""
        try:
            from fastvlm import FastVLM, FastVLMProcessor
            
            print(f"Loading FastVLM model from {self.model_path}...")
            self.model = FastVLM.from_pretrained(self.model_path).to(self.device)
            self.processor = FastVLMProcessor.from_pretrained(self.model_path)
            self.model.eval()
            print("Model loaded successfully.")
            
        except ImportError:
            raise ImportError(
                "FastVLM is not installed. Please install it from "
                "https://github.com/apple/ml-fastvlm"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load FastVLM model: {e}")
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess an image for the model.
        
        Args:
            image: Input image as a numpy array (BGR format from OpenCV)
            
        Returns:
            Preprocessed image tensor
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Process with FastVLM processor
        inputs = self.processor(images=pil_image, return_tensors="pt")
        return inputs.pixel_values.to(self.device)
    
    def is_ppt_slide(self, image: np.ndarray, threshold: float = 0.3) -> Tuple[bool, float, str]:
        """Check if an image contains PPT content.
        
        Args:
            image: Input image as a numpy array (BGR format from OpenCV)
            threshold: Confidence threshold for PPT detection
            
        Returns:
            tuple: (is_ppt, confidence, explanation)
        """
        try:
            # Prepare the prompt
            prompt = "What is shown in this image? Is this a slide or presentation content? " \
                    "Look for elements like text, bullet points, charts, or diagrams that are " \
                    "typical in presentation slides."
            
            # Preprocess image
            inputs = self.preprocess_image(image)
            
            # Generate description
            with torch.no_grad():
                outputs = self.model.generate(
                    pixel_values=inputs,
                    max_length=100,
                    num_beams=5,
                    no_repeat_ngram_size=2,
                    early_stopping=True
                )
            
            # Decode the output
            description = self.processor.decode(outputs[0], skip_special_tokens=True).lower()
            
            # Check for PPT indicators in the description
            confidence = self._calculate_ppt_confidence(description)
            is_ppt = confidence >= threshold
            
            return is_ppt, confidence, description
            
        except Exception as e:
            print(f"Error during PPT detection: {e}")
            return False, 0.0, f"Error: {str(e)}"
    
    def _calculate_ppt_confidence(self, text: str) -> float:
        """Calculate confidence that the text describes a PPT slide.
        
        Args:
            text: Generated description text
            
        Returns:
            float: Confidence score between 0 and 1
        """
        if not text:
            return 0.0
            
        # Check for PPT-related keywords
        text_lower = text.lower()
        keyword_matches = sum(1 for kw in self.ppt_keywords if kw in text_lower)
        
        # Calculate confidence based on keyword matches
        confidence = min(keyword_matches / len(self.ppt_keywords) * 1.5, 1.0)
        
        # Additional checks for slide-like content
        if 'slide' in text_lower or 'presentation' in text_lower:
            confidence = min(confidence + 0.2, 1.0)
            
        return confidence
    
    def extract_text_elements(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Extract text elements from an image that might be part of a slide.
        
        Args:
            image: Input image as a numpy array (BGR format from OpenCV)
            
        Returns:
            List of text elements with their positions and confidence
        """
        # This is a placeholder. In a real implementation, you would use OCR or similar
        # to extract text elements from the image.
        return []

def test_ppt_detection():
    """Test function for PPT detection."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test PPT detection')
    parser.add_argument('--model-path', required=True, help='Path to FastVLM model')
    parser.add_argument('--image', required=True, help='Path to test image')
    args = parser.parse_args()
    
    # Initialize detector
    detector = PPTDetector(model_path=args.model_path)
    
    # Load test image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image {args.image}")
        return
    
    # Check if it's a PPT slide
    is_ppt, confidence, description = detector.is_ppt_slide(image)
    
    print(f"Image: {args.image}")
    print(f"Description: {description}")
    print(f"Is PPT: {is_ppt}")
    print(f"Confidence: {confidence:.2f}")
    
    # Display the image
    cv2.imshow("Test Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_ppt_detection()
