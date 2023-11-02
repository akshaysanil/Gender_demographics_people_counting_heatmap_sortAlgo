import cv2
import numpy as np
from openvino.inference_engine import IECore

# Load the OpenVINO model
def gender_finding(img):
    model_xml = '/home/suraj/Documents/POC/poc_hmp_final/gender_model/model/age-gender-recognition-retail-0013.xml'
    model_bin = '/home/suraj/Documents/POC/poc_hmp_final/gender_model/model/age-gender-recognition-retail-0013.bin'

    ie = IECore()
    net = ie.read_network(model=model_xml, weights=model_bin)
    exec_net = ie.load_network(network=net, device_name='CPU')

    # Load and preprocess the image
    # image_path = img
    # image = cv2.imread(image_path)
    input_blob = next(iter(net.input_info))
    n, c, h, w = net.input_info[input_blob].input_data.shape
    input_image = cv2.resize(img, (w, h))
    input_image = input_image.transpose((2, 0, 1))
    input_image = input_image.reshape((n, c, h, w))

    # Perform inference
    output = exec_net.infer(inputs={input_blob: input_image})

    # Interpret the results (age and gender)
    # age = output['age_conv3'][0][0][0][0] * 100  # Age is predicted as a value between 0 and 1, so we scale it up to years
    gender_probs = output['prob'][0]
    # gender_probs = output[gender_output_blob][0]

    gender = 'Male' if gender_probs[1] > gender_probs[0] else 'Female'

    # print(f"Predicted Age: {age:.2f} years")
    print(f"Predicted Gender: {gender}")
    return gender
