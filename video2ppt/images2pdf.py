"""Convert a list of images to a single PDF file."""
import os
from typing import List, Union
from fpdf import FPDF


def images_to_pdf(outpath: str, images: List[Union[str, os.PathLike]], 
                 width: int = 1920, height: int = 1080) -> None:
    """Convert a list of images to a single PDF file.
    
    Args:
        outpath: Path to save the output PDF file
        images: List of image file paths to include in the PDF
        width: Width of the output PDF in points (default: 1920)
        height: Height of the output PDF in points (default: 1080)
    """
    pdf = FPDF()
    pdf.compress = False
    title_height = 20  # Reduced from 60 to make more space for the image
    
    # Calculate page size (swap width/height for landscape)
    size = (height + title_height, width)
    
    for image_path in images:
        try:
            # Add a new page for each image
            pdf.add_page(orientation='L', format=size, same=False)
            
            # Add image title
            pdf.set_font('helvetica', size=12)
            pdf.cell(
                w=400, 
                h=title_height, 
                txt=os.path.basename(str(image_path)), 
                border=0, 
                align='C',
                ln=2  # Move to next line after cell
            )
            
            # Add the image, leaving space for the title
            pdf.image(
                name=str(image_path), 
                x=0, 
                y=title_height + 5,  # Add small margin below title
                w=width, 
                h=height - title_height - 10,  # Adjust height for title and margin
                type='JPG' if str(image_path).lower().endswith('.jpg') else None
            )
            
        except Exception as e:
            print(f"Error adding image {image_path} to PDF: {e}")
    
    # Save the PDF
    pdf.output(outpath, "F")


# Alias for backward compatibility
images2pdf = images_to_pdf


if __name__ == '__main__':
    # Example usage
    test_images = ['data/frame00:00:01-0.jpg', 'data/frame00:00:06-0.56.jpg']
    images_to_pdf('data/test.pdf', test_images)