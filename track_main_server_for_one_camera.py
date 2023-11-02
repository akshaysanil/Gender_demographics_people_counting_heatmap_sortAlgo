# limit the number of cpus used by high performance libraries
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# server_ip = os.environ["SERVER_IP"]
# print('server ip ............',server_ip)

import sys
sys.path.insert(0, './yolov5')

import numpy as np
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, 
                                  check_imshow, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

from collections import deque

import time
from datetime import datetime
from heatmap import heatmap,heatmap_v2
from gender_model import main
import threading
from scipy.ndimage.filters import gaussian_filter
import matplotlib.cm as cm
from matplotlib import pyplot as plt
import requests
import imagezmq
import base64
from rtsp_stream_for_dashboard import video_stream,app

# server_ip = 


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

data = []
people_enter = {}
people_leave = {}
data_deque = {}
male_entering = {}
female_entering = {}
male_leaving = {}
female_leaving = {}
main_url = 'http://192.168.0.159:3000'

# line = [(608,515),(1223,513)] # line for downstairs office
# line = [(597,597),(1475,572)] # line for 1 st floor
# line = [(597,697),(1680,697)] #  line for 1st floor narrow space
line = [(0,540),(1920,540)] # half line
# line = [(550,640),(1420,640)] #  line



def push_counted_data_database(counted_data):
    servername = f"{main_url}/People_Count/Register"
    response = requests.post(servername, data=counted_data)
    # logger.info(violation_data)
    # print('response.status_code >>>>>>>>>>>>>>>>> : ',response.status_code )
    if response.status_code == 200:
        # logger.info("Data to be Pushed-=-=-=-=->")
        print('push_counted_data_database, pushed successfully',response.status_code)
    else:
        # logger.info('Failed to post data to the API.')
        print('push_counted_data_database, pusing failed',response.status_code)
    return

def push_camera_details_and_hmp_data_database(camera_details_and_hmp):
    servername = f"{main_url}/VA_API/Register"

    response = requests.post(servername,data = camera_details_and_hmp)
    print('response.status_code >>>>>>>>>>>>>>>>> : ',response.status_code )
    if response.status_code == 200 :
        print('push_camera_details_and_hmp_data_database, pushed successfully ',response.status_code)
    else:
        print('push_camera_details_and_hmp_data_database, pushing failed ',response.status_code)
    return

#------------to convert image to string------------#
def image_string_conv(img_raw):
    # img = cv2.imread(img_raw)
    _, buffer = cv2.imencode('.jpg', img_raw)
    image_string = base64.b64encode(buffer).decode('utf-8')
    return image_string







def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def get_direction(point1, point2):
    direction_str = ""

    # calculate y axis direction
    if point1[1] > point2[1]:
        direction_str += "South"
    elif point1[1] < point2[1]:
        direction_str += "North"
    else:
        direction_str += ""

    # calculate x axis direction
    if point1[0] > point2[0]:
        direction_str += "East"
    elif point1[0] < point2[0]:
        direction_str += "West"
    else:
        direction_str += ""
    return direction_str

def calculate_centroids(bbox_xyxy):
    centroids = []
    for box in bbox_xyxy:
        x1,y1,x2,y2 = box
        centroid_x = int((x1+x2)/2)
        centroid_y = int((y1+y2)/2)
        centroids.append((centroid_x,centroid_y))
    return centroids


def draw_boxes(im0,bboxes,obj_id,identities,names,offset=(0, 0)):
    cv2.line(im0, line[0], line[1], (46,162,112), 3)
    height, width, _ = im0.shape

    for key in list(data_deque):
        if key not in identities:
            data_deque.pop(key)

    for i , box in enumerate (bboxes):
        # print('bboxes-----',bboxes)
        x1,y1,x2,y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        center = (int((x2+x1)/ 2), int((y2+y1)/2))

        id = int(identities[i] if identities is not None else 0 )

        if obj_id[i] != 0:
            continue
    
        
        if id not in data_deque:
            data_deque[id] = deque(maxlen=64)
        obj_name = names[obj_id[i]]

        data_deque[id].appendleft(center)

        if len(data_deque[id]) >= 2:
            direction = get_direction(data_deque[id][0], data_deque[id][1])
            if intersect(data_deque[id][0], data_deque[id][1], line[0], line[1]):
                
                # headimage = im0[y1:y2, x1:x2]
                # result = main.gender_finding(headimage)

                cv2.line(im0, line[0], line[1], (255, 255, 255), 3)
                if "South" in direction:
                    people_enter[obj_name] = people_enter.get(obj_name, 0) + 1
                    headimage = im0[y1:y2, x1:x2]
                    result = main.gender_finding(headimage)
                    if result == 'Male':
                        male_entering[result] = male_entering.get(result,0) +1
                        print('no of male entering : ',male_entering)
                    else:
                        female_entering[result] = female_entering.get(result, 0) + 1
                        print('no of female entering : ',female_entering)

                if "North" in direction:
                    people_leave[obj_name] = people_leave.get(obj_name, 0) + 1
                    headimage = im0[y1:y2, x1:x2]
                    result = main.gender_finding(headimage)
                    if result == 'Male':
                        male_leaving[result] = male_leaving.get(result, 0) +1
                        print('no of male leaving : ',male_leaving)
                    else:
                        female_leaving[result] = female_leaving.get(result, 0) +1
                        print('no of female leaving : ',female_leaving)

        # draw trail
        # ---------------------------------------------integrated on dashboard / no need to show this ----------------------------------------#
        for i in range(1, len(data_deque[id])):
            # check if on buffer value is none
            if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                continue
            # generate dynamic thickness of trails
            thickness = int(np.sqrt(64 / float(i + i)) * 1.5)

            # draw trails
            cv2.line(im0, data_deque[id][i - 1], data_deque[id][i], [255,0,255], thickness)

        for idx, (key, value) in enumerate(people_enter.items()):
            # cnt_str1 = str(key) + ":" +str(value)
            cnt_str1 = str(value)
            cv2.line(im0, (20,60), (695,60), [85,45,255], 110)
            cv2.putText(im0, f'Total persons entering : {cnt_str1}', (11, 35), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)    
            cv2.putText(im0, f'Numbers of male entering : {sum(male_entering.values())}', (11, 75), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)    
            cv2.putText(im0, f'Numbers of female entering : {sum(female_entering.values())}', (11, 115), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)    
            # cv2.putText(im0, cnt_str1, (517, 35), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

        for idx, (key,value) in enumerate(people_leave.items()):
            # cnt_str = str(key) + ":" +str(value)
            cnt_str =  str(value)

            cv2.line(im0, (1290,60), (1800,60), [85,45,255], 110)
            cv2.putText(im0, f'Total persons leaving : {cnt_str}', (1281, 35), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(im0, f'Number of male leaving : {sum(male_leaving.values())}', (1281, 75), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(im0, f'Number of female leaving : {sum(female_leaving.values())}', (1281, 115), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
            # cv2.putText(im0, cnt_str, (517, 95), 0, 1, [255, 255, 255], thickness = 2, lineType = cv2.LINE_AA)
    
        total_people_inside = max(0,sum(people_enter.values()) - sum(people_leave.values()))

        print('total_people_inside : ',total_people_inside)

        print('no of person leaving :',people_leave)
        print('no of male leaving : ',male_leaving)
        print('no of female leaving : ',female_leaving)

        print('no of person entering :',people_enter)
        print('no of male entering : ',male_entering)
        print('no of female  entering: ', female_entering)
    
    #----------------------- TO PASS ALL THE COUNTING TO API ------------------------#
    people_entering = sum(people_enter.values())
    add_male_entering = sum(male_entering.values())
    add_female_entering = sum(female_entering.values())
    people_leaving = sum(people_leave.values())
    add_male_leaving = sum(male_leaving.values())
    add_female_leaving = sum(female_leaving.values())

    outputs = {"Camera_ID":"1   ",
               "people_enter":people_entering,
               "people_leave":people_leaving,
               "male_leaving":add_male_leaving,
               "female_leaving":add_female_leaving,
               "male_entering":add_male_entering,
               "female_entering":add_female_entering,
               }
    #"live_frame":im0
    # push_counted_data_database(outputs)
    # print('push_data_database : ',outputs)
    # return im0,people_enter,people_leave,male_leaving,female_leaving,male_entering,female_entering
    return im0,people_entering,add_male_entering,add_female_entering,people_leaving,add_male_leaving,add_female_leaving


def detect(opt):
    centroid_list = []
    start_time = time.time() *1000

    # interval of time for producing hmp
    intervel = 120000


    # ------------------- Getting RTSP Data from API and generating it into a FILE-----------------------#
    # only when you are using api 
    rtsp_list_url = "http://192.168.0.159:3000/VA_API/CameraList"

    response = requests.get(rtsp_list_url)
    rtsp_list = []
    all_camera_details = {}
    tobebroadcast = True
    if response.status_code == 200:
        data = response.json()
        port = 5555  # Add the initial port number here
        for camera in data['CameraList']:  # It should be 'CameraList', not 'CameraData'
            auth_key = camera['AuthKey']
            cam_rtsp = camera['RTSP_Link']
            cam_id = camera['Camera_ID']
            cam_location = camera['Location']
            cam_ip = camera['Camera_IP']
            flag = camera['Camera_Type']
            
            print('auth_key : ',auth_key)
            print('cam_rtsp : ',cam_rtsp)
            print('cam_id : ',cam_id)
            print('cam_location : ',cam_location)
            print('cam_ip : ',cam_ip)
            rtsp_list.append(cam_rtsp)
            all_camera_details[str(cam_id)] = {'Location': cam_location, 'RTSP_Link': cam_rtsp, 'AuthKey': auth_key}

            if tobebroadcast:
                sender = imagezmq.ImageSender(connect_to=f'tcp://*:{port}', REQ_REP=False)
                host_name = str(port)
                all_camera_details[str(cam_id)]["zmq_sender"] = sender
                all_camera_details[str(cam_id)]["zmq_host"] = host_name  # Corrected variable name
                port += 1


    # stream_to_dashboard,port = video_stream(cam_rtsp)
    # if stream_to_dashboard:
    #     url = f'http://192.168.0.189:5000/video'
    # else:
    #     url = None
    # video_stream(cam_rtsp)
    url = f'http://192.168.0.190:5000/video'
    # if __name__ == "__main__":
    #     app.run(host='0.0.0.0', port=5000)
        


    


    all_stream_filename = "./streams.txt"
    with open(all_stream_filename, 'w') as file:
        for cam_id in all_camera_details:
            file.write(str(all_camera_details[str(cam_id)]['RTSP_Link'] + '\n'))
    print('all_streams names :',all_stream_filename)    
    source_rtsp = './streams.txt'


    out, source, yolo_model, deep_sort_model, show_vid, save_vid, save_txt, imgsz, evaluate, half, project, name, exist_ok= \
        opt.output, opt.source, opt.yolo_model, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.imgsz, opt.evaluate, opt.half, opt.project, opt.name, opt.exist_ok
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt') or source_rtsp


    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(deep_sort_model,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA


    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder


    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        # dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        dataset = LoadStreams(source_rtsp, img_size=imgsz, stride=stride, auto=True)
        bs = len(dataset)  # batch_size
    else:       
        # dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=True)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(save_dir)) + '/' + txt_file_name + '.txt'

    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0

    
    for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
        t1 = time_sync()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if opt.visualize else False
        pred = model(img, augment=opt.augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam: 
                 # batch_size >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            s += '%gx%g ' % img.shape[2:]  # print string
            # if im0:
            annotator = Annotator(im0, line_width=2, pil=not ascii)
            w, h = im0.shape[1],im0.shape[0]

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                # if im0:
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()



                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                t4 = time_sync()
                # if im0_c:
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4

                # draw boxes for visualization


                if len(outputs) > 0:
                    identities = outputs[:, -2]
                    object_id = outputs[:, -1]
                    bbox_xyxy = outputs[:, :4]
                    height,width = im0.shape[:2]


                    centroids = calculate_centroids(bbox_xyxy)
                    centroid_list.extend(centroids)

                    current_time = time.time() *1000
                    total_time_taken = current_time - start_time
                    time_stamp = int(time.time())
                    formated_time = datetime.fromtimestamp(time_stamp).strftime('%Y_%m_%d__%H_%M_%S')

                    # im0, people_enter, people_leave, male_leaving, female_leaving, male_entering, female_entering = draw_boxes(im0, bbox_xyxy, object_id, identities, model.names)
                    im0,people_entering,add_male_entering,add_female_entering,people_leaving,add_male_leaving,add_female_leaving = draw_boxes(im0, bbox_xyxy, object_id, identities, model.names)

                    counted_Data = {"AuthKey":auth_key,
                                    "Camera_ID":cam_id,
                                    "people_enter":people_entering,
                                    "people_leave":people_leaving,
                                    "male_entering":add_male_entering,
                                    "female_entering":add_female_entering,
                                    "male_leaving":add_male_leaving,
                                    "female_leaving":add_female_leaving,
                                    "url":url}
                    push_counted_data_database(counted_Data)
        
                    if total_time_taken >= intervel:
                        start_time_hmp = time.time() * 1000
                        hmp_result = heatmap.heatmap(im0,centroid_list,height, width)


                        image_to_string_for_db = image_string_conv(hmp_result)
                        
                        end_time_hmp = time.time() * 1000
                        time_taken_hmp = end_time_hmp - start_time_hmp
                        print('time_taken_hmp :',time_taken_hmp)
                        cv2.imwrite(f'hmp_50per_intencity/{cam_id}{formated_time}.png',hmp_result)
                        start_time = current_time
                        centroid_list = []
                        
                        camera_details_and_hmp = {"Camera_IP":cam_ip,
                                                "Camera_ID": cam_id,
                                                "Location":  cam_location,
                                                "HeatMap_Image":  image_to_string_for_db,
                                                "AuthKey": auth_key}
                        push_camera_details_and_hmp_data_database(camera_details_and_hmp)
                        
                    else:
                        pass

                        for j,(output, conf) in enumerate(zip(outputs, confs)):
                            bboxes = output[0:4]
                            id = output[4]
                            cls = output[5]
                            # identities = 
                            c = int(cls)
                            label = f'{id} {names[c]} {conf:.2f}'
                            annotator.box_label(bboxes, label, color= colors(c,True))

                            if save_txt:
                                bbox_left = output[0]
                                bbox_top = output[1]
                                bbox_w = output[2] - output[0]
                                bbox_h = output[3] - output[1]
                                #write mot compliant results to file
                                with open(txt_path, 'a') as f:
                                    f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id , bbox_left,
                                                                bbox_top,bbox_w,bbox_h, -1,-1,-1,-1))
                LOGGER.info(f'{s}Done. YOLO:({t3 -t2:.3f}s), Deepsort:({t5 - t4:.3f}s)')

            else:
                deepsort.increment_ages()
                LOGGER.info('No detections')

            # Stream results
            im0 = annotator.result()
            if show_vid:
                global count
                color=(0,255,0)
                start_point = (0, h-350)
                end_point = (w, h-350)
                thickness = 3
                org = (150, 150)
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 3
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]

                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms deep sort update \
        per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        print('Results saved to %s' % save_path)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', nargs='+', type=str, default='models/crowdhuman_yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
    parser.add_argument('--source', type=str, default='videos/Traffic.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[480], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_false', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')

    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven

    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=10000, help='maximum detection per image')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    with torch.no_grad():
        detect(opt)