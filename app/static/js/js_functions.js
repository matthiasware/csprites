function updateSlider(label_id, slideAmount, prestring) {
    var sliderDiv = document.getElementById(label_id);
    sliderDiv.innerHTML = prestring+slideAmount;
}

function updateImageSize(label_id, slideAmount, prestring) {
    var sliderDiv = document.getElementById(label_id);
    sliderDiv.innerHTML = prestring+(2**slideAmount);
    document.getElementById("img_size").src = "../static/sizes/"+(2**slideAmount)+".png";
    adjustSliders();
}

function updateColormap(label_id, slideAmount, prestring) {
    var sliderDiv = document.getElementById(label_id);
    sliderDiv.innerHTML = prestring+slideAmount;
    document.getElementById("colormap").src = "../static/colors/"+slideAmount+".png";
    adjustSliders()
}

function updateRotation(label_id, slideAmount, prestring) {
    var sliderDiv = document.getElementById(label_id);
    sliderDiv.innerHTML = prestring+slideAmount;
    document.getElementById("rotations").src = "../static/rotations/gifs/"+slideAmount+".gif";
    adjustSliders()
}

function updateFillrate(label_id, slideAmount, prestring, adjusted) {
    var sliderDiv = document.getElementById(label_id);
    sliderDiv.innerHTML = prestring+slideAmount;

    var minLbl = document.getElementById("lbl_slider5");
    var maxLbl = document.getElementById("lbl_slider6");
    var minDiv = document.getElementById("slider5");
    var maxDiv = document.getElementById("slider6");

    if (minDiv.value>maxDiv.value) {
        if (adjusted==0){
            // minDiv was adjusted -> adapt maxDiv
            maxDiv.value = minDiv.value
            maxLbl.innerHTML = '<b>Maximum Fillrate: </b>' + maxDiv.value
        }else{
            // maxDiv was adjusted -> adapt minDiv
            minDiv.value = maxDiv.value
            minLbl.innerHTML = '<b>Minimum Fillrate: </b>' + minDiv.value
        }
    }
    adjustSliders();
}

function updateBgMode() {
    var nmb_field = document.getElementById("nmb_backgrounds")
    var lbl_field = document.getElementById("lbl_nmb_bg")

    if (nmb_field.disabled){
        // enable number selection
        nmb_field.disabled = false;
        lbl_field.style.color = "black";
    }else{
        // disable number selection
        nmb_field.disabled = true;
        lbl_field.style.color = "gray";
    }
    adjustSliders()
}


function assertSelectedShape() {
    var dates = $('[id^="cb"]');
    var arrayLength = dates.length;
    var checked = false
    for (var i = 0; i < arrayLength; i++) {
        checked = checked || dates[i].checked
    }
    if (!checked){
        dates[0].checked = true
    }
    adjustSliders();
}

function updatePosition() {
    var slider = document.getElementById("slider_pos")
    var label = document.getElementById("lbl_slider_pos");
    label.innerHTML = "<b>Positions: </b>"+slider.value;
    adjustSliders();
}

function updateScale() {
    var slider = document.getElementById("slider_scale")
    var label = document.getElementById("lbl_slider_scale");
    label.innerHTML = "<b>Scales: </b>"+slider.value;
    adjustSliders();
}

function updatePercentage() {
    var slider = document.getElementById("slider_scale")
    var label = document.getElementById("lbl_slider_scale");
    label.innerHTML = "<b>Scales: </b>"+slider.value;
}

function adjustSliders(edit_amount=false) {
    var slider_scale = document.getElementById("slider_scale");
    var label_scale = document.getElementById("lbl_slider_scale");
    var current_scale_value = slider_scale.value;

    var slider_pos = document.getElementById("slider_pos");
    var label_pos = document.getElementById("lbl_slider_pos");
    var current_pos_value = slider_pos.value;
    
    var min_fillrate = document.getElementById("slider5").value;
    var max_fillrate = document.getElementById("slider6").value;
    var img_size = 2**(document.getElementById("slider1").value);
    var min_mask_area = img_size**2 * min_fillrate;
    var max_mask_area = img_size**2 * max_fillrate;
    var nmb_scales = [];
    var mask_sizes = [];
    
    // iterate over selected shapes, find minimum amount of scales
    var cb_shapes = $('[id^="cb"]');
    var arrayLength = cb_shapes.length;
    let n_shapes = 0;

    let max_mask_size = 0;
    let min_mask_size = 0;

    for (var i = 0; i < arrayLength; i++) {
        if (cb_shapes[i].checked){
            let shape_min_mask_size = 1e9;
            n_shapes = n_shapes + 1;
            let shape_scales = possible_scales[cb_shapes[i].value]['area'];
            let shape_masks = possible_scales[cb_shapes[i].value]['mask_size'];
            for (var j = 0; j < shape_scales.length; j++) {
                if (shape_scales[j]>min_mask_area && shape_scales[j]<max_mask_area){
                    count = count + 1;
                    max_mask_size = Math.max(max_mask_size, shape_masks[j]);
                    shape_min_mask_size = Math.min(shape_min_mask_size, shape_masks[j]);
                }
            }
            min_mask_size = Math.max(min_mask_size, shape_min_mask_size);
        }
    }


    // now get number of scales > min_mask_size
    for (var i = 0; i < arrayLength; i++) {
        if (cb_shapes[i].checked){
            let shape_scales = possible_scales[cb_shapes[i].value]['area'];
            let shape_masks = possible_scales[cb_shapes[i].value]['mask_size'];
            var count = 0;
            for (var j = 0; j < shape_scales.length; j++) {
                if (shape_scales[j]>min_mask_area && shape_scales[j]<max_mask_area && shape_masks[j]>=min_mask_size){
                    count = count + 1;
                }
            }
            nmb_scales.push(count);
        }
    }

    
    // adjust scales
    var max_scales = Math.min(...nmb_scales);
    slider_scale.max = max_scales;
    slider_scale.value = Math.min(current_scale_value, max_scales);
    label_scale.innerHTML = "<b>Scales: </b>"+slider_scale.value;


    // adjust positions
    var max_positions = img_size - max_mask_size + 1;
    slider_pos.max = max_positions;
    slider_pos.value = Math.min(current_pos_value, max_positions);
    label_pos.innerHTML = "<b>Positions: </b>"+slider_pos.value;


    // adjust status information
    /**
     * n_masks = n_shapes * n_colors * n_angles * n_scales
     * n_states = n_masks * n_positions**2
     */

    var n_colors = document.getElementById("slider3").value;
    var n_angles = document.getElementById("slider4").value;
    var n_scales = slider_scale.value;
    var n_positions = slider_pos.value;

    var n_masks = n_shapes * n_colors * n_angles * n_scales;
    var n_states = n_masks * n_positions**2;
    
    var sample_field = document.getElementById("nmb_samples");
    var n_samples = sample_field.value;

    n_samples = Math.min(n_samples, n_states);
    if(!edit_amount){
        n_samples = n_states;
    }
    sample_field.value = n_samples;
    sample_field.max = n_states;


    var mem_usage = img_size**2 * 3 * n_samples * (10 ** (-9));
    mem_usage = Math.round(mem_usage *1e2)/1e2;
    document.getElementById("lbl_nmb_states").innerHTML = n_states;
    document.getElementById("lbl_nmb_masks").innerHTML = n_masks;
    document.getElementById("lbl_mem_usage").innerHTML = mem_usage;

    //var lbl_msg = document.getElementById("tmp_label");
    //lbl_msg.innerHTML = "states: "+n_states+" masks: "+n_masks;

    var btn_submit = document.getElementById("btn_submit");
    var mem_warning = document.getElementById("mem_warning");
    if (mem_usage>max_memory){
        btn_submit.disabled = true;
        mem_warning.style.display = "block";
    }else{
        btn_submit.disabled = false;
        mem_warning.style.display = "none";
    }
}






