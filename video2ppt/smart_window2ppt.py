"""Smart window monitor for extracting PPT slides using FastVLM."""

import argparse
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .window_capture import WindowCapture
from .fastvlm_detector import FastVLMDetector
from .images2pdf import images_to_pdf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SmartWindowMonitor:
    """Monitor a window and extract PPT slides using FastVLM for content detection."""
    
    def __init__(self, output_dir: str, model_path: str, interval: float = 1.0, 
                 similarity_threshold: float = 0.6, confidence_threshold: float = 0.3,
                 debug: bool = False, device: Optional[str] = None):
        """Initialize the smart window monitor.
        
        Args:
            output_dir: Directory to save extracted slides
            model_path: Path to FastVLM model directory
            interval: Time between captures in seconds
            similarity_threshold: Similarity threshold for detecting slide changes (0-1)
            confidence_threshold: Minimum confidence for PPT detection (0-1)
            debug: Enable debug mode (saves all frames)
            device: Device to run the model on ('cuda', 'mps', or 'cpu')
        """
        self.output_dir = Path(output_dir)
        self.interval = max(0.2, interval)  # Minimum 200ms between captures
        self.similarity_threshold = max(0.1, min(1.0, similarity_threshold))
        self.confidence_threshold = max(0.1, min(1.0, confidence_threshold))
        self.debug = debug
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize window capture
        self.capture = WindowCapture()
        
        # Initialize FastVLM detector
        logger.info(f"Loading FastVLM model from {model_path}...")
        self.detector = FastVLMDetector(model_path=model_path, device=device)
        
        # Store previous frame and its features
        self.prev_frame = None
        self.prev_features = None
        self.slide_count = 0
        self.frame_count = 0
        
        # Debug directory
        if self.debug:
            self.debug_dir = self.output_dir / "debug"
            self.debug_dir.mkdir(exist_ok=True)
            logger.info(f"Debug mode enabled. Debug frames will be saved to {self.debug_dir}")
        
        # Create slides directory
        self.slides_dir = self.output_dir / 'slides'
        self.slides_dir.mkdir(exist_ok=True)
    
    def select_window(self) -> None:
        """Interactively select a window to monitor."""
        print("Please select the window to monitor...")
        self.window_info = self.capture.interactive_select_window()
        print(f"Selected window: {self.window_info.get('title')}")
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a frame from the selected window.
        
        Returns:
            Captured frame as a numpy array, or None if capture failed
        """
        if not self.window_info:
            raise RuntimeError("No window selected")
        
        try:
            frame = self.capture.capture_window(
                window_id=self.window_info.get('id')
            )
            return frame
        except WindowCaptureError as e:
            print(f"Error capturing window: {e}")
            return None
    
    def _calculate_similarity(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate similarity between two frames using SSIM."""
        if frame1 is None or frame2 is None:
            return 0.0
            
        try:
            # Resize frames to the same dimensions if needed
            if frame1.shape != frame2.shape:
                h, w = min(frame1.shape[0], frame2.shape[0]), min(frame1.shape[1], frame2.shape[1])
                frame1 = cv2.resize(frame1, (w, h))
                frame2 = cv2.resize(frame2, (w, h))
            
            # Convert to grayscale
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            # Calculate SSIM
            from skimage.metrics import structural_similarity as ssim
            score = ssim(gray1, gray2, win_size=3, data_range=255)
            return float(score)
            
        except Exception as e:
            logger.warning(f"SSIM calculation failed: {e}")
            if self.debug:
                import traceback
                logger.debug(traceback.format_exc())
            return 0.0
    
    def _is_slide_transition(self, current_frame: np.ndarray) -> Tuple[bool, float]:
        """Check if the current frame represents a slide transition."""
        if self.prev_frame is None:
            return True, 0.0  # First frame is always considered a new slide
            
        # Calculate similarity with previous frame
        similarity = self._calculate_similarity(self.prev_frame, current_frame)
        is_transition = similarity < self.similarity_threshold
        
        return is_transition, similarity
    
    def _detect_ppt_content(self, frame: np.ndarray) -> Dict[str, Any]:
        """Detect if the frame contains PPT content."""
        try:
            is_ppt, confidence, description = self.detector.is_ppt_content(frame)
            return {
                'is_ppt': is_ppt,
                'confidence': confidence,
                'description': description,
                'error': False
            }
        except Exception as e:
            logger.error(f"Error in PPT detection: {e}")
            return {
                'is_ppt': False,
                'confidence': 0.0,
                'description': str(e),
                'error': True
            }
    
    def process_frame(self, frame: np.ndarray) -> bool:
        """Process a single frame and save if it's a new slide."""
        if frame is None or frame.size == 0:
            return False
            
        self.frame_count += 1
        is_saved = False
        
        try:
            # Check for slide transition
            is_transition, similarity = self._is_slide_transition(frame)
            
            # Only process further if this is a potential slide transition
            if is_transition:
                # Detect PPT content
                detection = self._detect_ppt_content(frame)
                
                if self.debug:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    debug_path = self.debug_dir / f"frame_{self.frame_count:04d}_sim{similarity:.2f}_conf{detection.get('confidence', 0):.2f}.jpg"
                    cv2.imwrite(str(debug_path), frame)
                    logger.debug(f"Saved debug frame to {debug_path}")
                
                # If this is a PPT slide, save it
                if detection.get('is_ppt', False):
                    logger.info(f"Detected PPT content (confidence: {detection.get('confidence', 0):.2f}): {detection.get('description', '')}")
                    is_saved = self._save_slide(frame, detection)
                    if is_saved:
                        self.prev_frame = frame.copy()
                        return True
            
            # If no slide was saved but we're in debug mode, save the frame anyway
            if self.debug and not is_saved and self.frame_count % 10 == 0:  # Save every 10th frame
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                debug_path = self.debug_dir / f"frame_{self.frame_count:04d}_debug.jpg"
                cv2.imwrite(str(debug_path), frame)
            
            return is_saved
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            if self.debug:
                import traceback
                logger.error(traceback.format_exc())
            return False
            
        
        # Update frames
    
    def _save_slide(self, frame: np.ndarray, detection: Dict[str, Any]) -> bool:
        """Save a slide to the output directory with metadata."""
        try:
            # Create filename based on timestamp and slide count
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"slide_{self.slide_count:04d}_{timestamp}.png"
            slide_path = self.slides_dir / filename
            
            # Save frame to file
            cv2.imwrite(str(slide_path), frame)
            logger.info(f"Saved slide to {slide_path}")
            
            # Save metadata
            metadata = {
                'slide_number': self.slide_count,
                'timestamp': timestamp,
                'filename': filename,
                'detection': {
                    'is_ppt': detection.get('is_ppt', False),
                    'confidence': detection.get('confidence', 0.0),
                    'description': detection.get('description', '')
                },
                'frame_count': self.frame_count
            }
            
            import json
            metadata_path = slide_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Update slide count
            self.slide_count += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving slide: {e}")
            if self.debug:
                import traceback
                logger.error(traceback.format_exc())
            return False
    
    def monitor_window(self, window_id: Optional[str] = None, duration: Optional[float] = None):
        """Monitor a window and extract PPT slides."""
        try:
            # List available windows if no window_id is provided
            if window_id is None:
                print("\nAvailable windows:")
                windows = self.capture.list_windows()
                for i, win in enumerate(windows):
                    print(f"{i}: {win.get('title', 'Untitled')} (PID: {win.get('pid', 'N/A')})")
                
                while True:
                    selection = input("\nSelect a window by number or enter window title (or 'q' to quit): ").strip()
                    if selection.lower() == 'q':
                        print("Operation cancelled by user")
                        return
                    
                    if not selection:
                        continue
                        
                    try:
                        # Try to get window by index
                        window_idx = int(selection)
                        if 0 <= window_idx < len(windows):
                            window_id = windows[window_idx]['title']
                            break
                        else:
                            print(f"Invalid window number. Please enter a number between 0 and {len(windows)-1}")
                    except ValueError:
                        # Use input as window title
                        window_id = selection
                        break
            
            print(f"\nMonitoring window: {window_id}")
            print(f"Output directory: {self.output_dir.absolute()}")
            print(f"Settings: interval={self.interval}s, similarity_threshold={self.similarity_threshold}, confidence_threshold={self.confidence_threshold}")
            print("\nPress 'q' in the preview window to stop monitoring...\n")
            
            start_time = time.time()
            last_log_time = start_time
            
            try:
                cv2.namedWindow("Preview (Press 'q' to quit)", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Preview (Press 'q' to quit)", 1024, 576)
                
                while True:
                    # Check duration
                    current_time = time.time()
                    if duration is not None and (current_time - start_time) > duration:
                        logger.info("Monitoring duration reached. Stopping...")
                        break
                    
                    # Log status every 10 seconds
                    if current_time - last_log_time > 10:
                        logger.info(f"Monitoring... Captured {self.slide_count} slides so far")
                        last_log_time = current_time
                    
                    try:
                        # Capture window
                        frame_start_time = time.time()
                        frame = self.capture.capture_window(window_id)
                        
                        if frame is None or frame.size == 0:
                            logger.warning("Failed to capture window")
                            time.sleep(1)
                            continue
                        
                        # Process frame
                        self.process_frame(frame)
                        
                        # Show preview
                        preview = cv2.resize(frame, (1024, 576))
                        
                        # Calculate processing time and FPS
                        process_time = time.time() - frame_start_time
                        fps = 1.0 / (process_time + 1e-6)
                        
                        # Display FPS and slide count
                        cv2.putText(preview, f"FPS: {fps:.1f} | Slides: {self.slide_count}", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow("Preview (Press 'q' to quit)", preview)
                        
                        # Check for quit key
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            logger.info("Stopped by user")
                            break
                        
                        # Calculate sleep time to maintain desired interval
                        elapsed = time.time() - frame_start_time
                        sleep_time = max(0, self.interval - elapsed)
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                        
                    except KeyboardInterrupt:
                        logger.info("Stopped by user")
                        break
                    except Exception as e:
                        logger.error(f"Error during monitoring: {e}")
                        if self.debug:
                            import traceback
                            logger.error(traceback.format_exc())
                        time.sleep(1)  # Prevent tight loop on errors
            
            finally:
                cv2.destroyAllWindows()
                logger.info(f"Monitoring stopped. Extracted {self.slide_count} slides to {self.output_dir.absolute()}")
        
        except Exception as e:
            logger.error(f"Error in monitor_window: {e}")
            if self.debug:
                import traceback
                logger.error(traceback.format_exc())
    
    def export_to_pdf(self, output_path: Optional[str] = None) -> str:
        """Export captured slides to a PDF file.
        
        Args:
            output_path: Path to save the PDF. If None, uses the output directory.
            
        Returns:
            Path to the generated PDF file.
        """
        if output_path is None:
            output_path = str(self.output_dir / 'presentation.pdf')
        
        # Get all slide images
        slide_files = sorted(self.slides_dir.glob('*.png'))
        
        if not slide_files:
            print("No slides found to export.")
            return ""
        
        # Convert to list of strings
        slide_paths = [str(f) for f in slide_files]
        
        # Create PDF
        images_to_pdf(output_path, slide_paths)
        print(f"Exported {len(slide_paths)} slides to {output_path}")
        
        return output_path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract PPT slides from a window using FastVLM"
    )
    
    # Required arguments
    parser.add_argument(
        "--model-path", "-m", 
        type=str, 
        required=True,
        help="Path to FastVLM model directory"
    )
    
    # Optional arguments
    parser.add_argument(
        "--window", "-w", 
        type=str, 
        help="Window title or ID to monitor (prompt if not provided)"
    )
    parser.add_argument(
        "--output", "-o", 
        type=str, 
        default="./slides",
        help="Output directory for extracted slides (default: ./slides)"
    )
    parser.add_argument(
        "--interval", "-i", 
        type=float, 
        default=1.0,
        help="Time between captures in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--similarity", "-s", 
        type=float, 
        default=0.6,
        help="Similarity threshold for slide change detection (0-1, default: 0.6)"
    )
    parser.add_argument(
        "--confidence", "-c", 
        type=float, 
        default=0.3,
        help="Minimum confidence for PPT detection (0-1, default: 0.3)"
    )
    parser.add_argument(
        "--duration", "-d", 
        type=float, 
        help="Maximum duration to monitor in seconds"
    )
    parser.add_argument(
        "--export-pdf", 
        action="store_true",
        help="Export slides to PDF when done"
    )
    parser.add_argument(
        "--pdf-name", 
        type=str, 
        default="presentation.pdf",
        help="Output PDF filename (default: presentation.pdf)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        choices=['auto', 'cuda', 'mps', 'cpu'],
        default='auto',
        help="Device to run the model on (default: auto)"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode (saves all frames and additional debug info)"
    )
    
    return parser.parse_args()


def main():
    """Command-line interface for smart window monitoring."""
    args = parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('ppt_extractor.log')
        ]
    )
    
    # Set device
    device = None
    if args.device != 'auto':
        device = args.device
    
    try:
        # Initialize monitor
        logger.info("Initializing PPT extractor...")
        monitor = SmartWindowMonitor(
            output_dir=args.output,
            model_path=args.model_path,
            interval=args.interval,
            similarity_threshold=args.similarity,
            confidence_threshold=args.confidence,
            debug=args.debug,
            device=device
        )
        
        # Start monitoring
        logger.info("Starting window monitoring...")
        monitor.monitor_window(window_id=args.window, duration=args.duration)
        
        # Export to PDF if requested
        if args.export_pdf and monitor.slide_count > 0:
            pdf_path = os.path.join(args.output, args.pdf_name)
            logger.info(f"Exporting {monitor.slide_count} slides to PDF: {pdf_path}")
            monitor.export_to_pdf(pdf_path)
        
        logger.info("PPT extraction completed successfully")
        return 0
    
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 0
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.debug:
            import traceback
            logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    main()
