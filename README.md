# Video Extractor

Extract content from videos and presentations using AI-powered detection. Supports PPT slides extraction from videos and screen recordings.

## Features

- Extract PPT slides from video files
- Capture slides directly from any window on your screen
- AI-powered content detection to identify PPT slides
- Export extracted slides as PDF
- Cross-platform support (Windows and macOS)
- Simple command-line interface

## install
``` shell
# install from pypi
pip install extract-video-ppt

# or local
python ./setup.py install

# or local user
python ./setup.py install --user
```

## Installation

### Prerequisites
- Python 3.8 or higher
- PyTorch (will be installed automatically)
- FFmpeg (for video processing)

### Install from PyPI
```bash
pip install extract-video-ppt
```

### Install from source
```bash
# Clone the repository
git clone https://github.com/yourusername/extract-video-ppt.git
cd extract-video-ppt

# Install with pip
pip install .

# Or install in development mode
pip install -e .
```

## Usage

### Extract PPT from Video File
```bash
# Basic usage
evp --similarity 0.6 --pdfname presentation.pdf --start_frame 0:00:09 --end_frame 00:30:00 ./output ./input_video.mp4

# With custom settings
evp --similarity 0.7 --pdfname my_presentation.pdf --start_frame 0:01:30 ./output ./another_video.mp4
```

### Extract PPT from Screen Capture
```bash
# Basic window capture (saves all frames)
evp-window --output ./slides --interval 1.0

# Smart window capture with AI detection (requires FastVLM model)
evp-smart-window --model-path /path/to/fastvlm/model --output ./slides --confidence 0.4

# With custom settings
evp-smart-window \
  --model-path /path/to/fastvlm/model \
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