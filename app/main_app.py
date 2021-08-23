from time import process_time
from dotted_dict import DottedDict
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file
import utils
import math
import threading
import time
import os
import io
import shutil

MAX_AGE = 500 # maximum age of files in seconds
app = Flask(__name__)
app.secret_key = "super secret key"

ds_creator = utils.DatasetCreator()

@app.route('/')
def home():
    clean_requests()
    return render_template('home.html')

@app.route('/select1', methods=['GET', 'POST'])
def select1():
    clean_requests()
    process_params, possible_scales = utils.get_initial_params()
    if request.method == 'POST':
        # Image Size
        value_slider_size = int(request.form.get("slider_img_size"))
        process_params.img_size = value_slider_size

        # Selected Shapes
        tmp = []
        for shape_name in process_params.shapes:
            x = request.form.get(f"shape_selection_{shape_name}")
            if x is not None:
                tmp.append(x)
        process_params.selected_shapes = tmp
        
        # Colors
        value_slider_colors = int(float(request.form.get("slider_colors")))
        process_params.colors = value_slider_colors

        # Rotations
        value_slider_rot = int(float(request.form.get("slider_rotations")))
        process_params.nmb_rotations = value_slider_rot

        # Background
        value_bg_select = request.form.get("bg_selection")
        process_params.bg_mode = value_bg_select

        value_nmb_bg_switch = request.form.get("switch_bg")
        value_nmb_bg = request.form.get("nmb_bg")

        if value_nmb_bg_switch is not None:
            value_nmb_bg = int(value_nmb_bg_switch)
        else:
            value_nmb_bg = int(value_nmb_bg)
        process_params.nmb_backgrounds = value_nmb_bg

        # Filling
        value_slider_min_fill = float(request.form.get("slider_min_fillrate"))
        process_params.min_fillrate = value_slider_min_fill
    
        value_slider_max_fill = float(request.form.get("slider_max_fillrate"))
        process_params.max_fillrate = value_slider_max_fill

        # Positions
        value_slider_positions = int(request.form.get("slider_pos"))
        process_params.positions = value_slider_positions

        # Scales
        value_slider_scales = int(request.form.get("slider_scale"))
        process_params.scales = value_slider_scales

        # Number of Samples
        value_n_samples = int(request.form.get("nmb_samples"))
        process_params.n_samples = value_n_samples
        
        # Dataset percentage
        value_test_perc = float(request.form.get("slider_test_perc"))
        process_params.test_perc = value_test_perc

        # Targets
        value_bbox = request.form.get("switch_bbox")=='0'
        process_params.target_bbox = value_bbox

        value_seg = request.form.get("switch_seg")=='0'
        process_params.target_seg = value_seg
        
        # n_states, n_masks, memory usage
        process_params = utils.recalculate_params(process_params)

        # save session parameters
        for k in process_params:
            session[k] = process_params[k]
        return redirect(url_for('create_dataset'))
    else:
        return render_template('select1.html', params = process_params, pos_scales=possible_scales)


@app.route('/create_ds', methods=['GET', 'POST'])
def create_dataset():
    process_params = DottedDict({k : session[k] for k in session})

    # ID from timestamp
    global ds_creator
    ds_creator.status = 'Initialization'
    request_id = str(time.time()).replace('.','-')
    th = threading.Thread(target=thread_func, args=(process_params,request_id))
    th.start()

    return render_template('create_ds.html', request_id=request_id)

def thread_func(process_params, request_id):
    global ds_creator
    try:
        ds_creator.create_dataset(process_params, request_id)
    except:
        print('Error: Could not create dataset!')

@app.route('/status')
def thread_status():
    global ds_creator
    state_dict = {
        'status' : ds_creator.status
    }
    return jsonify(state_dict)

@app.route('/<request_id>/download')
def ds_download(request_id):
    p = os.path.join('static','requests', request_id)
    name = [x for x in os.listdir(p) if x.endswith('.zip')][0]
    p_file = os.path.join(p, name)

    # read file as byte stream
    return_data = io.BytesIO()
    with open(p_file, 'rb') as fo:
        return_data.write(fo.read())
    # (after writing, cursor will be at last byte, so move it to start)
    return_data.seek(0)
    shutil.rmtree(p)

    return send_file(return_data, mimetype='application/zip', as_attachment=True, attachment_filename=name)

def clean_requests():
    t_now = time.time()
    p = os.path.join('static', 'requests')
    l = [(float(x.replace('-','.')), os.path.join(p,x)) for x in os.listdir(p)]
    for x, p in l:
        diff = t_now-x
        if diff > MAX_AGE:
            shutil.rmtree(p)

if __name__ == '__main__':
    app.run(debug=True)