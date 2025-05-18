# Video Extractor

Extract content from videos and presentations using AI-powered detection. This tool intelligently identifies and extracts PowerPoint slides from videos and screen recordings using advanced vision models.

## Features

- Extract PPT slides from video files
- Capture slides directly from any window on your screen
- AI-powered content detection to identify PPT slides using FastVLM
- Export extracted slides as PDF
- Cross-platform support (Windows and macOS)
- Simple command-line interface

## Installation

### Prerequisites
- Python 3.8 or higher
- PyTorch (will be installed automatically)
- FFmpeg (for video processing)

### Install from PyPI
```bash
pip install video-extractor
```

### Install from source
```bash
# Clone the repository
git clone https://github.com/chenxingqiang/video-extractor.git
cd video-extractor

# Install with pip
pip install .

# Or install in development mode
pip install -e .
```

### Download FastVLM Model (Optional)
For AI-powered slide detection, download the FastVLM model:

```bash
# Create model directory
mkdir -p checkpoints

# Download and extract model (1.5B parameter version recommended)
cd checkpoints
wget https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_1.5b_stage2.zip
unzip llava-fastvithd_1.5b_stage2.zip
cd ..
pip install -e .
```

## Usage

### Extract PPT from Video File

```shell
# Extract slides from a video file
video2ppt demovideo.mp4 --output ./slides

# With custom similarity threshold (0.0-1.0, higher means more strict)
video2ppt demovideo.mp4 --output ./slides --similarity 0.7

# Generate PDF after extraction
video2ppt demovideo.mp4 --output ./slides --export-pdf
```

### Extract PPT from Screen Capture
```bash
# Capture from a specific window
window2ppt --window "PowerPoint" --output ./captured_slides

# Capture with custom interval (in seconds)
window2ppt --window "Keynote" --output ./captured_slides --interval 2

# Capture with custom similarity threshold
window2ppt --window "PDF Viewer" --output ./captured_slides --similarity 0.8

# Export to PDF after capture (press Ctrl+C to stop capture and generate PDF)
window2ppt --window "PowerPoint" --output ./captured_slides --export-pdf

# Smart window capture with AI detection (requires FastVLM model)
smart_window2ppt --window "WeChat" --output ./slides --model-path ./checkpoints/llava-fastvithd_1.5b_stage2 --confidence 0.4

# With custom capture interval (in seconds)
smart_window2ppt --window "Zoom" --output ./slides --model-path ./checkpoints/llava-fastvithd_1.5b_stage2 --interval 2

# With debug mode for detailed logging
smart_window2ppt --window "Browser" --output ./slides --model-path ./checkpoints/llava-fastvithd_1.5b_stage2 --debug

# With custom settings
smart_window2ppt \
  --model-path demofastvlm/model \
  --output ./presentation_slides \
  --similarity 0.6 \
  --confidence 0.35 \
  --interval 0.5 \
  --export-pdf \
  --pdf-name my_presentation.pdf
```

## Parameters

### Video Extraction (evp)
- `--similarity`: Similarity threshold for detecting slide changes (0-1, default: 0.6)
- `--pdfname`: Output PDF filename (default: 'presentation.pdf')
- `--start_frame`: Start time in HH:MM:SS format (default: '00:00:00')
- `--end_frame`: End time in HH:MM:SS format (default: 'INFINITY')
- `output_dir`: Directory to save extracted slides
- `video_path`: Path to input video file

### Smart Window Capture (evp-smart-window)
- `--model-path`: Path to FastVLM model directory (required)
- `--output`: Output directory for extracted slides (default: './output')
- `--similarity`: Similarity threshold for detecting slide changes (0-1, default: 0.6)
- `--confidence`: Minimum confidence for PPT detection (0-1, default: 0.3)
- `--interval`: Time between captures in seconds (default: 1.0)
- `--duration`: Maximum duration to capture in seconds (optional)
- `--export-pdf`: Export slides to PDF when done
- `--pdf-name`: Output PDF filename (default: 'presentation.pdf')
- `--debug`: Enable debug mode (saves all frames)

## FastVLM Model

This tool uses the FastVLM model for intelligent PPT slide detection. You can download pre-trained models from the [ML-FastVLM repository](https://github.com/apple/ml-fastvlm).

### Example Model Download
```bash
# Download and extract the FastVLM model
wget https://github.com/apple/ml-fastvlm/releases/download/v1.0/fastvlm_1.5b_stage3.tar.gz
tar -xzf fastvlm_1.5b_stage3.tar.gz

# Use the model with the tool
evp-smart-window --model-path ./fastvlm_1.5b_stage3 --output ./slides
```

## Examples

### Extract slides from a video file
```bash
evp --similarity 0.65 --pdfname webinar_slides.pdf --start_frame 0:05:00 --end_frame 1:30:00 ./webinar_slides ./recorded_webinar.mp4
```

### Capture slides from a live presentation
```bash
# Start the capture before the presentation begins
evp-smart-window --model-path ./fastvlm_1.5b_stage3 --output ./meeting_notes --interval 0.5 --confidence 0.35

# After the presentation, press Ctrl+C to stop the capture
# The slides will be saved in the specified directory
```

## Testing

Following Test-Driven Development principles, here are the testing procedures for verifying the functionality of Video Extractor:

### 1. Testing Video Extraction

#### Test Setup:
- Prepare a test video containing presentation slides (use `demo/demo.mp4` if available)
- Create an output directory for extracted slides

#### Test Execution:
```bash
# Run basic extraction
video2ppt ./demo/demo.mp4 --output ./test_slides

# Verify extraction with different similarity thresholds
video2ppt ./demo/demo.mp4 --output ./test_slides_strict --similarity 0.8
video2ppt ./demo/demo.mp4 --output ./test_slides_lenient --similarity 0.4
```

#### Verification:
- Check that slides were extracted to the output directory
- Compare the number of slides extracted with different similarity thresholds
- Verify that the extracted slides accurately represent the content in the video

### 2. Testing Window Capture

#### Test Setup:
- Open a presentation in any application (PowerPoint, PDF viewer, etc.)
- Create an output directory for captured slides

#### Test Execution:
```bash
# Run basic window capture
window2ppt --window "Application Name" --output ./test_window_capture
```

#### Verification:
- Navigate through several slides in the presentation
- Press Ctrl+C to stop the capture
- Check that slides were captured to the output directory
- Verify that the captured slides match what was displayed

### 3. Testing AI-Powered Smart Window Capture

#### Test Setup:
- Ensure the FastVLM model is downloaded and available
- Open a video with presentation content in an application (e.g., WeChat, browser)
- Create an output directory for AI-detected slides

#### Test Execution:
```bash
# Run smart window capture with AI detection
smart_window2ppt --window "Application Name" --output ./test_ai_detection --model-path ./checkpoints/llava-fastvithd_1.5b_stage2 --debug
```

#### Verification:
- Play the video containing presentation slides
- Monitor the debug output to verify detection confidence
- Check that only slides (not other content) are being captured
- Press Ctrl+C to stop the capture
- Verify that the captured slides are high-quality and represent actual presentation content

## Troubleshooting

### Common Issues

1. **Model not found**
   - Make sure the model path is correct
   - Download the model files if they don't exist

2. **Performance issues**
   - Increase the `--interval` value to reduce CPU/GPU load
   - Use a smaller model if available
   - Close other resource-intensive applications

3. **Installation problems**
   - Make sure you have Python 3.8 or higher
   - Update pip: `pip install --upgrade pip`
   - Install build dependencies: `pip install wheel setuptools`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [ML-FastVLM](https://github.com/apple/ml-fastvlm) for the vision-language model
- [OpenCV](https://opencv.org/) for computer vision tasks
- [PyTorch](https://pytorch.org/) for deep learning framework