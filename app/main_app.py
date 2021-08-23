from flask import Flask, render_template, request
import utils
import math



app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/select1', methods=['GET', 'POST'])
def select1():
    process_params = utils.process_params
    possible_scales = utils.possible_scales
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
        value_seg = request.form.get("switch_seg")=='0'
        process_params.target_bbox = value_bbox
        process_params.target_seg = value_seg
        
        # n_states, n_masks, memory usage
        process_params = utils.recalculate_params(process_params)

        utils.create_dataset(process_params)

    return render_template('select1.html', params = process_params, pos_scales=possible_scales)


if __name__ == '__main__':
    app.run(debug=True)