"""
Extract PPT slides from a window capture (browser, app, etc.) by monitoring for changes.
"""
import os
import time
import click
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any

from .window_capture import WindowCapture, WindowCaptureError
from .compare import compareImg
from .images2pdf import images2pdf as _images2pdf

class WindowMonitor:
    """Monitor a window for changes and extract slides."""
    
    def __init__(self, output_dir: str, similarity_threshold: float = 0.6, 
                 capture_interval: float = 1.0, debug: bool = False):
        """Initialize the window monitor.
        
        Args:
            output_dir: Directory to save extracted slides
            similarity_threshold: Threshold for detecting slide changes (0-1)
            capture_interval: Time between captures in seconds
            debug: Enable debug mode (saves all frames for debugging)
        """
        self.output_dir = Path(output_dir)
        self.similarity_threshold = similarity_threshold
        self.capture_interval = capture_interval
        self.debug = debug
        
        self.window_capture = WindowCapture()
        self.window_info: Optional[Dict[str, Any]] = None
        self.last_frame: Optional[np.ndarray] = None
        self.slide_count = 0
        self.frame_count = 0
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.slides_dir = self.output_dir / 'slides'
        self.slides_dir.mkdir(exist_ok=True)
        
        if self.debug:
            self.debug_dir = self.output_dir / 'debug'
            self.debug_dir.mkdir(exist_ok=True)
    
    def select_window(self) -> None:
        """Interactively select a window to monitor."""
        print("Please select the window to monitor...")
        self.window_info = self.window_capture.interactive_select_window()
        print(f"Selected window: {self.window_info.get('title')}")
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a frame from the selected window.
        
        Returns:
            Captured frame as a numpy array, or None if capture failed
        """
        if not self.window_info:
            raise RuntimeError("No window selected")
        
        try:
            frame = self.window_capture.capture_window(
                window_id=self.window_info.get('id')
            )
            return frame
        except WindowCaptureError as e:
            print(f"Error capturing window: {e}")
            return None
    
    def process_frame(self, frame: np.ndarray) -> bool:
        """Process a captured frame and save if it's a new slide.
        
        Args:
            frame: Captured frame to process
            
        Returns:
            bool: True if a new slide was detected and saved, False otherwise
        """
        self.frame_count += 1
        
        # Save debug frame if enabled
        if self.debug:
            debug_path = self.debug_dir / f"frame_{self.frame_count:04d}.png"
            cv2.imwrite(str(debug_path), frame)
        
        # Skip if this is the first frame
        if self.last_frame is None:
            self.last_frame = frame
            return False
        
        # Compare with previous frame
        similarity = compareImg(frame, self.last_frame)
        
        # If significant change detected, save as new slide
        if similarity < self.similarity_threshold:
            self.slide_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            slide_path = self.slides_dir / f"slide_{self.slide_count:03d}_{timestamp}.png"
            
            # Save the slide
            cv2.imwrite(str(slide_path), frame)
            print(f"New slide detected! Saved as {slide_path.name}")
            
            # Update last frame
            self.last_frame = frame
            return True
        
        return False
    
    def monitor(self, duration: Optional[float] = None) -> None:
        """Monitor the selected window for changes.
        
        Args:
            duration: Duration to monitor in seconds. If None, monitor until interrupted.
        """
        if not self.window_info:
            self.select_window()
        
        print(f"Monitoring window: {self.window_info.get('title')}")
        print("Press Ctrl+C to stop monitoring...")
        
        start_time = time.time()
        
        try:
            while True:
                # Check if duration has been exceeded
                if duration is not None and (time.time() - start_time) > duration:
                    print(f"Monitoring completed after {duration} seconds.")
                    break
                
                # Capture and process frame
                frame = self.capture_frame()
                if frame is not None:
                    self.process_frame(frame)
                
                # Wait for next capture
                time.sleep(self.capture_interval)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user.")
        except Exception as e:
            print(f"Error during monitoring: {e}")
        finally:
            print(f"\nExtracted {self.slide_count} slides to {self.slides_dir}")
    
    def export_pdf(self, output_path: Optional[str] = None) -> str:
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
        _images2pdf(output_path, slide_paths)
        print(f"Exported {len(slide_paths)} slides to {output_path}")
        
        return output_path

@click.command()
@click.option('--output', '-o', default='./output', 
              help='Output directory for extracted slides and PDF')
@click.option('--similarity', '-s', default=0.6, type=float,
              help='Similarity threshold for detecting slide changes (0-1)')
@click.option('--interval', '-i', default=1.0, type=float,
              help='Time between captures in seconds')
@click.option('--duration', '-d', type=float,
              help='Duration to monitor in seconds (default: until interrupted)')
@click.option('--debug', is_flag=True, help='Enable debug mode (saves all frames)')
@click.option('--export-pdf', is_flag=True, help='Export slides to PDF when done')
@click.option('--pdf-name', default='presentation.pdf', help='Name of the output PDF file')
def main(output: str, similarity: float, interval: float, 
         duration: Optional[float], debug: bool, export_pdf: bool, pdf_name: str):
    """Extract PPT slides from a window by monitoring for changes."""
    try:
        # Initialize monitor
        monitor = WindowMonitor(
            output_dir=output,
            similarity_threshold=similarity,
            capture_interval=interval,
            debug=debug
        )
        
        # Start monitoring
        monitor.monitor(duration=duration)
        
        # Export to PDF if requested
        if export_pdf:
            pdf_path = os.path.join(output, pdf_name)
            monitor.export_pdf(pdf_path)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()
