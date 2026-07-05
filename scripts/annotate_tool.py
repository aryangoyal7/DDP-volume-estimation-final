import gradio as gr
import json
import os
from pathlib import Path
from PIL import Image

# Use the first image from each pile
IMG_PATHS = {
    "pile1": "solid_waste_dataset2_iitb_site_pho/pile1/IMG_7975.jpg",
    "pile2": "solid_waste_dataset2_iitb_site_pho/pile2/IMG_7983.jpg",
    "pile3": "solid_waste_dataset2_iitb_site_pho/pile3/IMG_7999.jpg"
}

CALIBRATION_FILE = "configs/box_calibration.json"
if os.path.exists(CALIBRATION_FILE):
    with open(CALIBRATION_FILE, "r") as f:
        polygon_data = json.load(f)
else:
    polygon_data = {}

def get_image(pile_name):
    # Load the image
    img_path = IMG_PATHS[pile_name]
    if os.path.exists(img_path):
        return Image.open(img_path)
    return None

def extract_points(evt: gr.SelectData, pile_name, current_points):
    # evt.index is (x, y)
    try:
        points = json.loads(current_points) if current_points else []
    except:
        points = []
        
    points.append([evt.index[0], evt.index[1]])
    
    # Save if we hit 4 points
    if len(points) == 4:
        polygon_data[pile_name] = points
        with open(CALIBRATION_FILE, "w") as f:
            json.dump(polygon_data, f, indent=4)
        return json.dumps(points), f"Saved 4 points for {pile_name}! You can move to the next pile."
    elif len(points) > 4:
        # Reset if they click again after 4
        points = [[evt.index[0], evt.index[1]]]
        return json.dumps(points), "Reset points. Please click the 4 corners again."
        
    return json.dumps(points), f"Clicked {len(points)}/4 points."

with gr.Blocks(title="Annotate White Box Corners") as demo:
    gr.Markdown("# 📦 White Box Annotator\nSelect the pile below, then click the **4 corners** of the inside white box floor (clockwise or counter-clockwise). It will auto-save to `configs/box_calibration.json` once you click 4 times.")
    
    with gr.Row():
        pile_dropdown = gr.Dropdown(choices=["pile1", "pile2", "pile3"], label="Select Pile", value="pile1")
        status_text = gr.Textbox(label="Status", value="Select a pile and click 4 corners on the image.", interactive=False)
    
    current_points = gr.Textbox(visible=False, value="[]")
    
    # Image component where user can click
    img_view = gr.Image(value=get_image("pile1"), label="Click 4 corners here")
    
    # Update image when dropdown changes
    pile_dropdown.change(
        fn=lambda p: (get_image(p), "[]", f"Loaded {p}. Waiting for 4 corner clicks..."),
        inputs=[pile_dropdown],
        outputs=[img_view, current_points, status_text]
    )
    
    # Handle clicks
    img_view.select(
        fn=extract_points,
        inputs=[pile_dropdown, current_points],
        outputs=[current_points, status_text]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
