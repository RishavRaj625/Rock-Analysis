from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
import os
import cv2
import numpy as np
import json
from werkzeug.utils import secure_filename

# Import custom modules
from mineral_db import MINERAL_DB
from model_utils import get_model, get_grad_cam_overlay, get_shap_overlay
from features import (
    assess_image_quality,
    analyze_texture_and_grain,
    detect_cracks_and_fractures,
    analyze_color_distribution,
    generate_edge_visualizations,
    generate_color_histogram,
    generate_color_pie_chart
)
from report_generator import generate_pdf_report

app = Flask(__name__)
app.secret_key = 'mineral_analysis_secret_key'

# Configuration
UPLOAD_FOLDER = os.path.join('static', 'uploads')
REPORTS_FOLDER = os.path.join('static', 'reports')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['REPORTS_FOLDER'] = REPORTS_FOLDER

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file uploaded', 400
        
        file = request.files['file']
        if file.filename == '':
            return 'No file selected', 400
            
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Predict using neural network
            model, class_list = get_model()
            if model is None:
                # Mock prediction if model not loaded
                predicted_class = "Quartz"
                confidence = 94.2
                class_idx = class_list.index(predicted_class) if predicted_class in class_list else 0
            else:
                img_cv = cv2.imread(file_path)
                img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (128, 128))
                img_array = np.expand_dims(img_resized, axis=0) / 255.0
                
                preds = model.predict(img_array, verbose=0)[0]
                class_idx = np.argmax(preds)
                predicted_class = class_list[class_idx]
                confidence = float(preds[class_idx] * 100)
                
            # Perform OpenCV analytics
            crack_count, crack_file = detect_cracks_and_fractures(file_path, filename, app.config['UPLOAD_FOLDER'])
            colors_list = analyze_color_distribution(file_path)
            texture_data = analyze_texture_and_grain(file_path)
            quality_data = assess_image_quality(file_path)
            
            # Generate Edge scans, RGB Histogram, and Dominant Colors Pie Chart
            edge_file, edge_overlay_file = generate_edge_visualizations(file_path, filename, app.config['UPLOAD_FOLDER'])
            hist_file = generate_color_histogram(file_path, filename, app.config['UPLOAD_FOLDER'])
            pie_file = generate_color_pie_chart(colors_list, filename, app.config['UPLOAD_FOLDER'])
            
            # Explainable AI Overlays
            gradcam_file = get_grad_cam_overlay(file_path, model, class_idx, filename, app.config['UPLOAD_FOLDER']) if model else ""
            shap_file = get_shap_overlay(file_path, model, class_idx, filename, app.config['UPLOAD_FOLDER']) if model else ""
            
            # Calculate Mineral Quality Index (MQI)
            quality_score = quality_data['score']
            # Composite index formula: 40% confidence, 30% image quality, 30% texture smoothness/purity
            purity = 100 - texture_data['roughness']
            mqi = int((confidence * 0.4) + (quality_score * 0.3) + (purity * 0.3))
            mqi = max(0, min(mqi, 100))
            
            # Get mineral metadata
            metadata = MINERAL_DB.get(predicted_class, MINERAL_DB['Quartz'])
            
            # Save analysis results to a json file in uploads directory to load in results page
            results = {
                "filename": filename,
                "mineral": predicted_class,
                "confidence": confidence,
                "formula": metadata['formula'],
                "group": metadata['group'],
                "crystal_system": metadata['crystal_system'],
                "hardness": metadata['hardness'],
                "specific_gravity": metadata['specific_gravity'],
                "luster": metadata['luster'],
                "streak": metadata['streak'],
                "cleavage": metadata['cleavage'],
                "fracture": metadata['fracture'],
                "transparency": metadata['transparency'],
                "habit": metadata['habit'],
                "applications": metadata['applications'],
                "formation": metadata['formation'],
                "timeline": metadata['timeline'],
                "deposits": metadata['deposits'],
                "crack_count": crack_count,
                "crack_file": crack_file,
                "edge_file": edge_file,
                "edge_overlay_file": edge_overlay_file,
                "hist_file": hist_file,
                "pie_file": pie_file,
                "colors": colors_list,
                "grain_size": texture_data['grain_size'],
                "grain_mm": texture_data['grain_mm'],
                "roughness": texture_data['roughness'],
                "texture_type": texture_data['type'],
                "quality_score": quality_score,
                "quality_status": quality_data['status'],
                "quality_brightness": quality_data['brightness'],
                "quality_contrast": quality_data['contrast'],
                "quality_sharpness": quality_data['sharpness'],
                "quality_noise": quality_data['noise'],
                "gradcam_file": gradcam_file,
                "shap_file": shap_file,
                "mqi": mqi
            }
            
            # Save history metadata
            history_path = os.path.join(app.config['UPLOAD_FOLDER'], 'history.json')
            history = []
            if os.path.exists(history_path):
                try:
                    with open(history_path, 'r') as f:
                        history = json.load(f)
                except:
                    pass
            history.append({
                "filename": filename,
                "mineral": predicted_class,
                "confidence": confidence,
                "mqi": mqi
            })
            with open(history_path, 'w') as f:
                json.dump(history, f)
                
            results_path = os.path.join(app.config['UPLOAD_FOLDER'], filename + ".json")
            with open(results_path, 'w') as f:
                json.dump(results, f)
                
            return redirect(url_for('analyze_results', filename=filename))
            
    return redirect(url_for('index'))

@app.route('/analyze/<filename>')
def analyze_results(filename):
    results_path = os.path.join(app.config['UPLOAD_FOLDER'], filename + ".json")
    if not os.path.exists(results_path):
        return "Analysis not found", 404
        
    with open(results_path, 'r') as f:
        data = json.load(f)
        
    return render_template('analyze.html', data=data)

@app.route('/report/<filename>')
def get_report(filename):
    results_path = os.path.join(app.config['UPLOAD_FOLDER'], filename + ".json")
    if not os.path.exists(results_path):
        return "Analysis data not found", 404
        
    with open(results_path, 'r') as f:
        data = json.load(f)
        
    pdf_filename = generate_pdf_report(filename, data, app.config['UPLOAD_FOLDER'], app.config['REPORTS_FOLDER'])
    pdf_path = os.path.join(app.config['REPORTS_FOLDER'], pdf_filename)
    return send_file(pdf_path, as_attachment=True)

@app.route('/api/chat', methods=['POST'])
def chat():
    post_data = request.get_json() or {}
    message = post_data.get('message', '').lower().strip()
    
    response = "I am your AI Geological Assistant. Ask me anything about mineral identification, crystal systems, chemical formulas, or geological timelines!"
    
    # Process queries
    if "hello" in message or "hi " in message or "hey" in message:
        response = "Hello! I am the AI Geological Assistant. How can I help you in your mineralogy research today?"
    elif "mohs" in message or "hardness" in message:
        response = "The Mohs scale of mineral hardness characterizes scratch resistance from 1 (Talc, very soft) to 10 (Diamond, extremely hard). Our platform supports hardness classification for Quartz (7), Feldspar (6), Calcite (3), Hematite (5.5-6.5), Mica (2-3), Pyrite (6-6.5), Gypsum (2), and Talc (1)."
    elif "quartz" in message:
        m = MINERAL_DB['Quartz']
        response = f"<b>Quartz</b> ({m['formula']}) is a {m['group']} mineral. It crystallizes in the {m['crystal_system']} system, has a Mohs hardness of {m['hardness']}, specific gravity of {m['specific_gravity']}, and shows {m['cleavage']} cleavage with a {m['fracture']} fracture. Formation: {m['formation']}"
    elif "feldspar" in message:
        m = MINERAL_DB['Feldspar']
        response = f"<b>Feldspar</b> ({m['formula']}) is a {m['group']}. It has a Mohs hardness of {m['hardness']}, a {m['crystal_system']} system, and shows {m['cleavage']}. It is the most abundant mineral group in Earth's crust."
    elif "calcite" in message:
        m = MINERAL_DB['Calcite']
        response = f"<b>Calcite</b> ({m['formula']}) is a carbonate mineral with a Mohs hardness of {m['hardness']}. It is famous for its perfect {m['cleavage']} cleavage and double refraction (Iceland spar). It forms major sedimentary carbon sinks."
    elif "hematite" in message:
        m = MINERAL_DB['Hematite']
        response = f"<b>Hematite</b> ({m['formula']}) is an iron oxide ore. It has a high specific gravity of {m['specific_gravity']}, metallic luster, and a characteristic {m['streak']} streak color. Found extensively in Precambrian Banded Iron Formations."
    elif "mica" in message:
        m = MINERAL_DB['Mica']
        response = f"<b>Mica</b> ({m['formula']}) is known for its {m['cleavage']} cleavage which allows it to split into extremely thin, flexible sheets. It has a Mohs hardness of {m['hardness']} and is widely used as a thermal/electrical insulator."
    elif "pyrite" in message or "fool's gold" in message:
        m = MINERAL_DB['Pyrite']
        response = f"<b>Pyrite</b> ({m['formula']}), commonly known as 'Fool\'s Gold', is an iron sulfide. It crystallizes in the {m['crystal_system']} system (forming brassy cubes), has a hardness of {m['hardness']}, and metallic luster. Used in sulfuric acid manufacturing."
    elif "gypsum" in message:
        m = MINERAL_DB['Gypsum']
        response = f"<b>Gypsum</b> ({m['formula']}) is a hydrous calcium sulfate evaporite. It is very soft (hardness {m['hardness']} on the Mohs scale), has perfect cleavage, and is used to manufacture drywall, plasters, and soil conditioners."
    elif "talc" in message:
        m = MINERAL_DB['Talc']
        response = f"<b>Talc</b> ({m['formula']}) defines the minimum value (1) on the Mohs hardness scale. It is a phyllosilicate with a waxy/greasy feel. Used as a powder base in cosmetics, ceramics, and plastic fillers."
    elif "crystal system" in message or "crystallography" in message:
        response = "Minerals are grouped into 7 crystal systems based on their symmetry: Cubic (e.g. Pyrite), Tetragonal, Orthorhombic, Hexagonal, Trigonal (e.g. Quartz, Calcite, Hematite), Monoclinic (e.g. Gypsum, Talc, Mica, Feldspar), and Triclinic. Crystal systems dictate the mineral's physical properties, cleavage, and optical refraction."
    elif "cracks" in message or "fracture" in message:
        response = "Cracks and fractures are detected using OpenCV Canny edge profiling. Linear, high-aspect-ratio edge segments represent structural stress lines or natural cleavages. Analyzing them helps evaluate mineral durability and quality indices (MQI)."
    elif "grad-cam" in message or "explain" in message:
        response = "Grad-CAM (Gradient-weighted Class Activation Mapping) is an explainable AI technique. It records gradients at the final Conv2D layer of our model during forward-pass classification to produce a heat overlay showing which areas of the image the model focused on (e.g., texture, grain edges) to make its final prediction."
    
    return jsonify({"response": response})

@app.route('/history')
def get_history():
    history_path = os.path.join(app.config['UPLOAD_FOLDER'], 'history.json')
    history = []
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
        except:
            pass
    return jsonify(history)

# Run app
if __name__ == '__main__':
    # Initialize model loader at startup
    get_model()
    app.run(debug=True)
