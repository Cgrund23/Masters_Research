import cv2
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

# Load and preprocess the image
def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    
    # Threshold the image to get a binary mask
    _, binary = cv2.threshold(gray, 105, 255, cv2.THRESH_BINARY_INV)
    plt.imshow(binary,cmap = 'gray')
    plt.show()
    return binary

# Extract the contours (representing the track layout)
def extract_contours(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Convert the contours into a path format for the SDF
def contours_to_path(contours):
    path_points = []
    for contour in contours:
        for point in contour:
            x, y = point[0]
            path_points.append((x, y))
    plt.plot(path_points)
    plt.show()
    return path_points

# Generate an SDF file with the extracted path and width
def generate_sdf(path_points, output_file, track_width=5.0):
    # Create an SDF XML tree
    sdf = ET.Element('sdf', version='1.6')

    # Define a simple ground plane
    model = ET.SubElement(sdf, 'model', name='track_model')
    ET.SubElement(model, 'static').text = 'true'
    
    # Create link for the track
    link = ET.SubElement(model, 'link', name='track_link')
    
    # Visual representation
    visual = ET.SubElement(link, 'visual', name='track_visual')
    geometry = ET.SubElement(visual, 'geometry')
    polyline = ET.SubElement(geometry, 'polyline')

    # Add the width of the track
    width = ET.SubElement(polyline, 'height')
    width.text = str(track_width)  # Set the track width (in meters or the desired unit)

    # Define the track's polyline path in the SDF
    for x, y in path_points:
        point = ET.SubElement(polyline, 'point')
        point.text = f'{x} {y} 0'  # The '0' represents the z-coordinate, assuming it's flat
    
    # Write the SDF to a file
    tree = ET.ElementTree(sdf)
    tree.write(output_file, encoding='utf-8', xml_declaration=True)

# Main function
def image_to_sdf(image_path, sdf_output_path):
    # # Preprocess the image to binary
    # binary_image = preprocess_image(image_path)
    
    # # Extract the contours (representing the track)
    # contours = extract_contours(binary_image)
    
    # # Convert the contours into a path format
    # path_points = contours_to_path(contours)
    path_points = [(0,0),(5,2),(10,2),]
    # Generate the SDF file
    generate_sdf(path_points, sdf_output_path)

# Example usage
image_path = 'track_image.png'  # Replace with the path to your track image
sdf_output_path = 'track.sdf'
image_to_sdf(image_path, sdf_output_path)

print(f"SDF file has been generated and saved to {sdf_output_path}")