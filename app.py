import os
import json
import cv2
import openai
import glob
import shutil
import subprocess
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename

########################################
# Flask App Setup
########################################
app = Flask(__name__, template_folder="templates", static_folder="static")

# You can set your openai api key overhere
os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = os.getenv("OPENAI_API_KEY")

# Yolo object detection model
MODEL_WEIGHTS = "best.pt"

########################################
# YOLO + LLM Drone Landing Functions
########################################

def run_yolo_inference(model_weights, image_path, conf=0.3):
    """
    1. Runs YOLO from the CLI using subprocess with your specified command.
    2. YOLO saves results (including the .jpg with drawn boxes and .txt labels)
       to runs/detect/predictX/.
    3. Finds the corresponding .txt file for the image, reads lines,
       and returns them in the YOLO format:
         class x_center y_center width height
       (all normalized to 0..1).
    """
    command = [
        "yolo",
        "task=detect",
        "mode=predict",
        f"model={model_weights}",
        f"source={image_path}",
        "imgsz=640",
        f"conf={conf}",
        "save_txt=True",
        "device=cpu",
        "show_conf=False",
    ]

    # Run the YOLO CLI
    subprocess.run(command, check=True)

    # Find the most recent "runs/detect/predict*" folder.
    run_folders = sorted(glob.glob("runs/detect/predict*"), key=os.path.getmtime)
    if not run_folders:
        print("No YOLO run folders found.")
        return [], None  # Return empty bboxes, no interim path

    latest_run_folder = run_folders[-1]

    # The predicted image with YOLO-drawn bboxes:
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    yolo_drawn_image = os.path.join(latest_run_folder, base_name + ".jpg")

    # The YOLO labels:
    label_file = os.path.join(latest_run_folder, "labels", base_name + ".txt")

    # If no .txt file, YOLO found no objects
    if not os.path.exists(label_file):
        print(f"No label file found => no detections for {base_name}")
        return [], yolo_drawn_image

    # Read lines from the .txt file
    with open(label_file, "r") as f:
        lines = f.read().strip().splitlines()

    return lines, yolo_drawn_image


def parse_yolo_predictions(prediction_lines, img_width, img_height):
    """
    Convert YOLO prediction lines (class, x_center, y_center, w, h)
    from normalized to pixel-based coordinates (x1, y1, x2, y2).
    Return a list of dicts: {class_id, x1, y1, x2, y2}.
    """
    bboxes = []
    for line in prediction_lines:
        cls_str, x_c_str, y_c_str, w_str, h_str = line.strip().split()
        
        class_id = int(cls_str)
        x_c, y_c, w, h = map(float, [x_c_str, y_c_str, w_str, h_str])
        
        # Convert to pixel coords
        box_width  = w  * img_width
        box_height = h  * img_height
        box_xc     = x_c * img_width
        box_yc     = y_c * img_height
        
        x1 = box_xc - (box_width  / 2)
        y1 = box_yc - (box_height / 2)
        x2 = box_xc + (box_width  / 2)
        y2 = box_yc + (box_height / 2)
        
        bboxes.append({
            "class_id": class_id,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        })
    return bboxes


def draw_bounding_box(image, box, color=(0, 255, 0), thickness=2):
    """
    Draw bounding box in-place on image using OpenCV.
    box is a dict with x1, y1, x2, y2 in pixel coords.
    """
    x1, y1 = int(box["x1"]), int(box["y1"])
    x2, y2 = int(box["x2"]), int(box["y2"])
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return image


def choose_landing_spot_with_chatgpt(bboxes, img_width, img_height):
    """
    Calls OpenAI's Chat API to determine the best landing spot, returning a bounding box.
    If a fire is detected (class=1) per your rules, we won't even call this function.
    """
    # Make sure the API key is set:
    if not openai.api_key:
        print("Warning: OpenAI API key not set.")
        return None

    system_message = (
        "You are a drone-landing assistant. You know bounding boxes for:"
        " 0=animal, 1=fire, 2=helipad, 3=person."
        " Your job is to decide on the best landing spot."
    )

    user_message = f"""
    We have these YOLO bounding box detection results for an image of size {img_width}x{img_height}:
    {bboxes}

    Rules:
    1. If any 'helipad' (class=2) exists, choose the largest one as a landing spot.
    2. If there is a fire, you do NOT return a bounding box. (But we won't call this if we detect fire.)
    3. Otherwise, choose the farthest point from detected humans or animals for landing. 
    4. Return a bounding box (about 60x60 pixels).

    IMPORTANT:
    - Return ONLY valid JSON in this format (no extra keys, no extra text):
    {{
        "x1": <float>,
        "y1": <float>,
        "x2": <float>,
        "y2": <float>
    }}
    
    - Do not include any markdown or additional explanations outside the JSON.
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",  # or 'gpt-4' if you have access
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.2
        )
        chatgpt_reply = response.choices[0].message.content
        print("DEBUG reply:", chatgpt_reply)


        # Attempt to parse the JSON box
        chosen_box = json.loads(chatgpt_reply)
        if all(k in chosen_box for k in ["x1", "y1", "x2", "y2"]):
            return {
                "x1": float(chosen_box["x1"]),
                "y1": float(chosen_box["y1"]),
                "x2": float(chosen_box["x2"]),
                "y2": float(chosen_box["y2"])
            }
        else:
            return None
    except Exception as e:
        print(f"Error with LLM: {e}")
        return None


def main_drone_landing_flow(model_weights, image_path, interim_path, final_path):
    """
    1) YOLO Inference: YOLO draws bounding boxes on the detection image in runs/detect/predictX/.
    2) Copy YOLO's drawn image to `interim_path` (so the user can see the interim result).
    3) Parse YOLO bounding boxes from the .txt file.
    4) If any 'fire' (class=1), create landing_output.txt and skip final bounding box.
    5) Otherwise, use ChatGPT to choose final bounding box and draw it on the *original* image => `final_path`.
    6) Return status about whether a landing spot was chosen or if a fire was found.
    """

    # 1) YOLO Inference
    yolo_labels, yolo_drawn_path = run_yolo_inference(model_weights, image_path)
    
    # 2) Copy YOLO’s drawn image to our static interim path (if it exists).
    if yolo_drawn_path and os.path.exists(yolo_drawn_path):
        shutil.copyfile(yolo_drawn_path, interim_path)

    # 3) Parse YOLO predictions
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Could not read image at {image_path}")
    img_height, img_width = original_image.shape[:2]
    
    bboxes = parse_yolo_predictions(yolo_labels, img_width, img_height)

    # 4) Check for fire
    fire_detected = any(b["class_id"] == 1 for b in bboxes)
    if fire_detected:
        # Write a text file to indicate the presence of fire
        with open("landing_output.txt", "w") as f:
            f.write("FIRE DETECTED. LANDING IS NOT POSSIBLE.")
        print("Fire detected. 'landing_output.txt' generated. No final bounding box.")
        
       
        cv2.imwrite(final_path, original_image)
        return "fire"

    # 5) If no fire, proceed with LLM-based decision
    chosen_spot = choose_landing_spot_with_chatgpt(bboxes, img_width, img_height)
    
    # 6) Draw final bounding box if we have one
    final_image = original_image.copy()
    if chosen_spot:
        draw_bounding_box(final_image, chosen_spot, color=(0, 255, 0), thickness=3)
        cv2.imwrite(final_path, final_image)
        return "landing_chosen"
    else:
        # No valid spot from ChatGPT
        cv2.imwrite(final_path, final_image)
        return "no_spot"


########################################
# Flask Routes
########################################
@app.route("/", methods=["GET", "POST"])
def index():
    interim_url = None
    final_url = None
    no_spot_message = None
    
    if request.method == "POST":
        # 1) Get the file from the request
        if 'input_image' not in request.files:
            return redirect(request.url)
        
        file = request.files['input_image']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            filename = secure_filename(file.filename)
            upload_path = os.path.join("static", filename)
            file.save(upload_path)  # Save uploaded image to static folder

            # 2) Run the main YOLO + LLM flow
            interim_path = os.path.join("static", "interim.jpg")
            final_path = os.path.join("static", "final.jpg")
            
            result_status = main_drone_landing_flow(
                MODEL_WEIGHTS,
                upload_path,
                interim_path,
                final_path
            )
            
            # 3) Prepare variables for template
            interim_url = url_for('static', filename="interim.jpg")
            final_url = url_for('static', filename="final.jpg")
            
            if result_status == "fire":
                no_spot_message = "FIRE DETECTED. LANDING IS NOT POSSIBLE. Please move a couple of kilometers before trying to land again"
            elif result_status == "no_spot":
                no_spot_message = "No valid landing spot found."
            elif result_status == "landing_chosen":
                no_spot_message = None  # Means we found a landing spot

    return render_template("index.html",
                           interim_url=interim_url,
                           final_url=final_url,
                           no_spot_message=no_spot_message)

if __name__ == "__main__":
    # You can specify a different host or port if needed
    app.run(debug=True, host="0.0.0.0", port=5000)
