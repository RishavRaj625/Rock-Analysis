import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from io import BytesIO
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import cv2
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
from collections import defaultdict
import base64

# Color analysis helper functions
def get_color_name_and_rgb(rgb):
    """
    Get color name and its true RGB value
    """
    color_map = {
        'Red': (255, 0, 0),
        'Green': (0, 255, 0),
        'Blue': (0, 0, 255),
        'Yellow': (255, 255, 0),
        'Cyan': (0, 255, 255),
        'Magenta': (255, 0, 255),
        'White': (255, 255, 255),
        'Black': (0, 0, 0),
        'Gray': (128, 128, 128),
        'Orange': (255, 165, 0),
        'Purple': (128, 0, 128),
        'Brown': (139, 69, 19),
        'Pink': (255, 182, 193),
        'Beige': (245, 245, 220),
        'Navy': (0, 0, 128),
        'Gold': (255, 215, 0),
        'Silver': (192, 192, 192),
        'Maroon': (128, 0, 0),
        'Olive': (128, 128, 0),
        'Teal': (0, 128, 128),
        'Light Blue': (173, 216, 230),
        'Dark Green': (0, 100, 0),
        'Dark Red': (139, 0, 0),
        'Light Gray': (211, 211, 211)
    }
    
    min_dist = float('inf')
    closest_color_name = None
    closest_color_rgb = None
    
    for color_name, color_rgb in color_map.items():
        dist = distance.euclidean(rgb, color_rgb)
        if dist < min_dist:
            min_dist = dist
            closest_color_name = color_name
            closest_color_rgb = color_rgb
    
    return closest_color_name, closest_color_rgb

def determine_optimal_clusters(pixels_scaled, max_clusters=15):
    """
    Determine the optimal number of color clusters using elbow method
    """
    inertias = []
    K = range(1, max_clusters + 1)
    
    for k in K:
        kmeans = KMeans(n_clusters=k, n_init=10)
        kmeans.fit(pixels_scaled)
        inertias.append(kmeans.inertia_)
    
    # Calculate the rate of change
    differences = np.diff(inertias)
    rates_of_change = np.diff(differences)
    
    # Find the elbow point
    elbow_point = np.argmin(np.abs(rates_of_change)) + 2
    
    return min(elbow_point + 1, max_clusters)

def merge_similar_colors(colors, proportions, color_names):
    """
    Merge similar colors and sum their proportions
    """
    merged_colors = defaultdict(float)
    color_to_rgb = {}
    
    # Sum proportions for identical colors
    for color, prop, name in zip(colors, proportions, color_names):
        color_tuple = tuple(map(int, color))
        merged_colors[name] += prop
        color_to_rgb[name] = color_tuple
    
    # Convert back to lists
    merged_names = list(merged_colors.keys())
    merged_proportions = list(merged_colors.values())
    merged_rgb = [color_to_rgb[name] for name in merged_names]
    
    # Sort by proportion
    sorted_indices = np.argsort(merged_proportions)[::-1]
    
    return (np.array(merged_rgb)[sorted_indices], 
            np.array(merged_proportions)[sorted_indices], 
            np.array(merged_names)[sorted_indices])

def get_dynamic_colors(image):
    """
    Get precise color distribution using dynamic number of clusters
    """
    # Reshape the image
    pixels = image.reshape(-1, 3)
    
    # Sample pixels for faster processing
    samples = min(50000, len(pixels))
    pixels_sample = pixels[np.random.choice(len(pixels), samples, replace=False)]
    
    # Scale features
    scaler = StandardScaler()
    pixels_scaled = scaler.fit_transform(pixels_sample)
    
    # Determine optimal number of clusters
    n_colors = determine_optimal_clusters(pixels_scaled)
    
    # Apply K-means clustering with optimal clusters
    kmeans = KMeans(n_clusters=n_colors, n_init=10)
    kmeans_labels = kmeans.fit_predict(pixels_scaled)
    
    # Get cluster centers and convert back to original scale
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    centers = np.clip(centers, 0, 255)
    
    # Calculate proportions
    proportions = np.bincount(kmeans_labels) / len(kmeans_labels)
    
    # Sort colors by proportion
    sorted_indices = np.argsort(proportions)[::-1]
    sorted_colors = centers[sorted_indices]
    sorted_proportions = proportions[sorted_indices]
    
    # Map to nearest named colors
    named_colors = []
    true_rgb_values = []
    for color in sorted_colors:
        name, rgb = get_color_name_and_rgb(color)
        named_colors.append(name)
        true_rgb_values.append(rgb)
    
    # Merge similar colors
    return merge_similar_colors(true_rgb_values, sorted_proportions, named_colors)

def load_and_preprocess(uploaded_file):
    """
    Modified to work with Streamlit's uploaded file
    """
    # Read the uploaded file as bytes
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    uploaded_file.seek(0)  # Reset file pointer
    
    if image is None:
        raise ValueError("Could not load image")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply bilateral filter to reduce noise while preserving edges
    denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Apply contrast enhancement
    lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l,a,b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    return enhanced

def create_enhanced_pie_chart(colors, percentages, color_names):
    """
    Create an enhanced pie chart with correct color representation
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare colors for plotting (normalize to 0-1 range)
    plot_colors = [np.array(color)/255 for color in colors]
    
    # Create custom labels
    labels = [f'{name}\n({percentages[i]:.1f}%)' 
             for i, name in enumerate(color_names)]
    
    # Create pie chart
    patches, texts, autotexts = plt.pie(
        percentages,
        labels=labels,
        colors=plot_colors,
        autopct='',
        pctdistance=0.85,
        startangle=90,
        labeldistance=1.1
    )
    
    plt.setp(texts, size=10)
    plt.title('Color Distribution Analysis', pad=20, size=14, weight='bold')
    
    # Add a white circle at the center for donut effect
    center_circle = plt.Circle((0,0), 0.70, fc='white')
    fig.gca().add_artist(center_circle)
    
    # Add legend
    legend_labels = [f'{name}: RGB{tuple(map(int, color))}' 
                    for name, color in zip(color_names, colors)]
    plt.legend(patches, legend_labels,
              title="Color Information",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))
    
    plt.axis('equal')
    return fig

def create_color_histogram(colors, percentages, color_names):
    """
    Create a histogram of color distribution
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare colors for plotting (normalize to 0-1 range)
    plot_colors = [np.array(color)/255 for color in colors]
    
    # Create bars
    bars = plt.bar(range(len(colors)), percentages, color=plot_colors)
    
    # Customize the plot
    plt.title('Color Distribution Histogram', pad=20, size=14, weight='bold')
    plt.xlabel('Colors')
    plt.ylabel('Percentage (%)')
    
    # Rotate x-axis labels for better readability
    plt.xticks(range(len(colors)), color_names, rotation=45, ha='right')
    
    # Add percentage labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def create_3d_scatter(colors, percentages, color_names):
    """
    Create a 3D scatter plot showing RGB color distribution
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalize colors to 0-1 range for plotting
    plot_colors = [np.array(color)/255 for color in colors]
    
    # Extract RGB values
    r = [color[0] for color in colors]
    g = [color[1] for color in colors]
    b = [color[2] for color in colors]
    
    # Create scatter plot with size proportional to percentage
    sizes = percentages * 50  # Scale factor for better visibility
    scatter = ax.scatter(r, g, b, c=plot_colors, s=sizes, alpha=0.6)
    
    # Add labels
    for i in range(len(colors)):
        ax.text(r[i], g[i], b[i], f'{color_names[i]}\n({percentages[i]:.1f}%)',
                horizontalalignment='center', verticalalignment='bottom')
    
    # Set labels and title
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    ax.set_title('3D RGB Color Distribution', pad=20, size=14, weight='bold')
    
    # Set axis limits
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_zlim(0, 255)
    
    plt.tight_layout()
    return fig

def create_color_relationship_graph(colors, percentages, color_names):
    """
    Create a graph showing relationships between colors based on RGB distance
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate distances between colors
    n_colors = len(colors)
    distances = np.zeros((n_colors, n_colors))
    
    for i in range(n_colors):
        for j in range(n_colors):
            distances[i,j] = np.sqrt(np.sum((np.array(colors[i]) - np.array(colors[j]))**2))
    
    # Normalize distances for plotting
    distances = distances / distances.max()
    
    # Create a relationship graph
    plt.imshow(distances, cmap='YlOrRd_r')
    plt.colorbar(label='Color Similarity (normalized)')
    
    # Add labels
    plt.xticks(range(n_colors), color_names, rotation=45, ha='right')
    plt.yticks(range(n_colors), color_names)
    
    # Add title
    plt.title('Color Relationship Graph', pad=20, size=14, weight='bold')
    
    # Add percentage annotations
    for i in range(n_colors):
        for j in range(n_colors):
            plt.text(j, i, f'{distances[i,j]:.2f}',
                    ha='center', va='center',
                    color='black' if distances[i,j] > 0.5 else 'white')
    
    plt.tight_layout()
    return fig

def save_plot_to_bytes(fig):
    """
    Convert matplotlib figure to bytes for PDF inclusion
    """
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)
    return buf.getvalue()

def create_pdf_report(image_data, colors, percentages, color_names, figs):
    """
    Create and return PDF report as bytes
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    heading_style = styles['Heading2']
    normal_style = styles['Normal']
    
    color_style = ParagraphStyle(
        'ColorStyle',
        parent=styles['Normal'],
        spaceAfter=12,
        spaceBefore=12,
        leading=16
    )
    
    content = []
    
    # Add title and timestamp
    content.append(Paragraph("Color Analysis Report", title_style))
    content.append(Spacer(1, 20))
    content.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    content.append(Spacer(1, 20))
    
    # Add original image
    content.append(Paragraph("Original Image", heading_style))
    content.append(Spacer(1, 10))
    img_stream = BytesIO(image_data)
    img = Image(img_stream)
    img.drawWidth = 400
    img.drawHeight = 300
    content.append(img)
    content.append(Spacer(1, 20))
    
    # Add color analysis summary
    content.append(Paragraph("Color Analysis Summary", heading_style))
    content.append(Spacer(1, 10))

    for name, color, percentage in zip(color_names, colors, percentages):
        content.append(Paragraph(
            f"{name} (RGB{tuple(map(int, color))}): {percentage:.1f}%",
            color_style
        ))
    
    content.append(Spacer(1, 20))
    
    # Add visualizations
    content.append(Paragraph("Visualizations", heading_style))
    content.append(Spacer(1, 10))
    
    viz_titles = [
        "Color Distribution (Donut Chart)",
        "Color Distribution Histogram",
        "3D RGB Color Distribution",
        "Color Relationship Graph"
        "Edge Detection"
    ]
    
    for title, fig in zip(viz_titles, figs.values()):
        content.append(Paragraph(title, heading_style))
        content.append(Spacer(1, 10))
        
        img_data = save_plot_to_bytes(fig)
        img = Image(BytesIO(img_data))
        img.drawWidth = 450
        img.drawHeight = 350
        content.append(img)
        content.append(Spacer(1, 20))
    
    doc.build(content)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

def save_plot_to_bytes(fig):
    """
    Convert matplotlib figure to bytes
    """
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)
    return buf.getvalue()

def detect_rock_edges(image_path):
    """
    Detect edges in rock image and return visualization figures
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load image")
    
    # Convert to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale for processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Enhance image
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # Detect edges using multiple methods
    sigma = 0.33
    median = np.median(enhanced)
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    edges_canny = cv2.Canny(enhanced, lower, upper)
    
    # Sobel Edge Detection
    sobelx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
    edges_sobel = np.sqrt(sobelx*2 + sobely*2)
    edges_sobel = np.uint8(edges_sobel / edges_sobel.max() * 255)
    
    # Combine edges
    edges = cv2.addWeighted(edges_canny, 0.7, edges_sobel, 0.3, 0)
    
    # Clean up edges
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Create overlay
    overlay = image_rgb.copy()
    overlay[edges > 0] = [255, 0, 0]  # Red color for edges
    
    # Blend with original image
    result = cv2.addWeighted(image_rgb, 0.7, overlay, 0.3, 0)
    
    # Create visualization figure
    fig = plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.imshow(image_rgb)
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(edges, cmap='gray')
    plt.title('Edges')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(result)
    plt.title('Edge Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    
    return edges, result, fig

def main():
    st.set_page_config(page_title="Color Analysis Tool", layout="wide")
    st.title("Image Color Analysis Tool")
    
    uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Display original image
        st.subheader("Original Image")
        st.image(uploaded_file, use_column_width=True)
        
        try:
            # Process the image
            processed_image = load_and_preprocess(uploaded_file)
            colors, percentages, color_names = get_dynamic_colors(processed_image)
            percentages = percentages * 100
            
            # Display color information
            st.subheader("Detected Colors")
            col1, col2 = st.columns(2)
            
            with col1:
                for name, color, percentage in zip(color_names, colors, percentages):
                    st.write(f"{name} (RGB{tuple(map(int, color))}): {percentage:.1f}%")
            
            # Create all visualizations
            figs = {}
            
            st.subheader("Color Distribution (Donut Chart)")
            figs['donut'] = create_enhanced_pie_chart(colors, percentages, color_names)
            st.pyplot(figs['donut'])
            
            st.subheader("Color Distribution Histogram")
            figs['histogram'] = create_color_histogram(colors, percentages, color_names)
            st.pyplot(figs['histogram'])
            
            st.subheader("3D RGB Color Distribution")
            figs['scatter'] = create_3d_scatter(colors, percentages, color_names)
            st.pyplot(figs['scatter'])
            
            st.subheader("Color Relationship Graph")
            figs['relationship'] = create_color_relationship_graph(colors, percentages, color_names)
            st.pyplot(figs['relationship'])
            
            # st.subheader("Detect Rock Edges")
            # figs['edges'] = detect_rock_edges(colors, percentages, color_names)
            # st.pyplot(figs['edges'])
            
            # Generate PDF report
            if st.button("Generate PDF Report"):
                pdf_bytes = create_pdf_report(
                    uploaded_file.getvalue(),
                    colors,
                    percentages,
                    color_names,
                    figs
                )
                
                # Create download button
                b64_pdf = base64.b64encode(pdf_bytes).decode()
                href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="color_analysis_report.pdf">Download PDF Report</a>'
                st.markdown(href, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()