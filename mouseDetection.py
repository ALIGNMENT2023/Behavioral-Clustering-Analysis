import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['CUDA_VISIBLE_DEVICES']='2' 
        
# from OCRNetwork import ocrmodel
import cv2
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
from scipy.io import loadmat, savemat
from yolov4.tf import YOLOv4
import tensorflow as tf
from tqdm import tqdm

yolo = YOLOv4()
# yolo.classes = "./../yolov4/coco.names"
yolo.classes = "D:/data/git/darknet/rodentDetection/obj.names"
yolo.make_model()
# yolo.load_weights("./../yolov4/yolov4.weights", weights_type="yolo")
yolo.load_weights(
    "D:/data/git/darknet/rodentDetection/yolov4-obj_last.weights", weights_type="yolo")

model_name = ''# user defines, like '20210123-134230'
saved_model_path = "D:/data/git/mouseTouch/ocr/saved_models/{}".format(model_name)
ocrmodel = tf.keras.models.load_model(saved_model_path) 

def mouseDetection(filePath='S08-20210119-12',gpuID=1,camType=3):   
    # import os
    # os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    # os.environ['CUDA_VISIBLE_DEVICES']=str(gpuID) 
            
    # # from OCRNetwork import ocrmodel
    # import cv2
    # from datetime import datetime
    # from matplotlib import pyplot as plt
    # import numpy as np
    
    # from yolov4.tf import YOLOv4
    # import tensorflow as tf
    # from tqdm import tqdm
    
    # yolo = YOLOv4()
    # # yolo.classes = "./../yolov4/coco.names"
    # yolo.classes = "D:/data/git/darknet/data/obj.names"
    # yolo.make_model()
    # # yolo.load_weights("./../yolov4/yolov4.weights", weights_type="yolo")
    # yolo.load_weights(
    #     "D:/data/git/darknet/backup/yolov4-obj_last.weights", weights_type="yolo")
    
    # model_name = '20210123-134230'
    # saved_model_path = "D:/data/git/mouseTouch/ocr/saved_models/{}".format(model_name)
    # ocrmodel = tf.keras.models.load_model(saved_model_path)

    # camType = 3
    # filePath = 'S08-20210119-12'
    analysisfolder = 'D:/data/AnalysisPKU'
    filename = os.path.split(filePath)[1]
    filename = os.path.splitext(filename)[0]
    if filename[0] == 'S' or filename[0] == 's':
        setupFolder = filename[0:3]
        parentfolder = filename[4:12]
    else:
        setupFolder = 'siat'
        parentfolder = filename[0:8]
    subfolder = 'A'+filename
    folderpath = os.path.join(analysisfolder, setupFolder, parentfolder, subfolder)
    if camType<3:
        matpath = os.path.join(folderpath, 'timeYolo'+str(camType+2)+'.mat')
        txtpath = os.path.join(folderpath, 'timeYolo'+str(camType+2)+'.txt')
    else:
        matpath = os.path.join(folderpath, 'timeYolo'+str(camType)+'.mat')
        txtpath = os.path.join(folderpath, 'timeYolo'+str(camType)+'.txt')
    if os.path.exists(txtpath):
        os.remove(txtpath)
    prefix = 'YiV'
    if camType == 1:
        video2read = prefix+filename+'-car_1.mp4'
    elif camType == 2:
        video2read = prefix+filename+'-car_2.mp4'
    elif camType == 3:
        video2read = prefix+filename+'-Origin1.mp4'
    elif camType == 4:
        video2read = prefix+filename+'-Origin2.mp4'
    vidpath = os.path.join(folderpath, video2read)
    print(vidpath)
    # cap = cv2.VideoCapture(0)
    # vidpath = '/home/xing/下载/labelled/YiV20200606-01-L01.mp4'
    # vid2write='/home/xing/下载/labelled/YiV20200606-01-L01-model{}.mp4'.format(model_name)
    if not os.path.exists(vidpath):
        return 0,0
    cap = cv2.VideoCapture(vidpath)
    folder_path = 'D:/data/Pictures/failframe/yolov4'
    file_path, file_extension = os.path.splitext(vidpath)
    file_extension = file_extension.replace('.', '_')
    file_path += file_extension
    video_name_ext = os.path.basename(file_path)
    desired_img_format = '.jpg'
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # videoWriter = cv2.VideoWriter(vid2write, cv2.VideoWriter_fourcc('M','P','E','G'), fps, (int(width),int(height)))
    
    # if os.path.exists(txtpath):
        # if camType==3:
        #     matpath=os.path.join(folderpath, filename+'-frameTime.mat')
        # elif camType==4:
        #     matpath=os.path.join(folderpath, filename+'-frameTimeL.mat')
        # vidData = loadmat(matpath)
        # tv = vidData['tv']
        # if len(tv)==n_frames:
        #     cap.release()
        #     return fps,0
    if len(filename)==15:
        date = filename[4:12]
    elif len(filename)==11:
        date = filename[:8]
    date1Sum = np.sum(np.int32(np.array(list(date)) == '1'))
    dateShift = 149-3*date1Sum
    
    brightArea = np.zeros((540,960))
    brightArea[120:420,480:900] = 1
            
    ilast = -1
    camModel = 1
    timeyolo = []
    frames2pick = tqdm(range(0, n_frames))
    # frames2pick = tqdm(range(0, 1000))
    fcnBegin = datetime.now()
    for i in frames2pick:
    # for i in range(n_frames):
    
        ret, frame = cap.read()
        if not ret:
            break
        
        # ------ OCR for time reading ------    
        img = cv2.resize(frame, (960, 540))
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        gray = np.multiply(gray,brightArea)
        lumi = np.int0(gray.sum()/brightArea.sum())
        
        if camModel == 1:
            img2 = img[493:518, 22:274, :]
            # img2 = frame[498:523,684:936,:]
        else:
            # img2 = img[498:523, 690:942, :]
            img2 = img[498:523, 685:937, :]
    
        timeShift2 = 0
        y2_prediction = 0
        hms = ''
        splitImage2 = np.zeros((28, 0, 3))
    
        for kk2 in range(6):
            if y2_prediction == 1:
                timeShift2 = timeShift2+3
            timeImage2 = img2[:, dateShift+14*kk2+8*(kk2//2)-timeShift2:dateShift+14*(kk2+1)+8*(kk2//2)-timeShift2, :]
            timeImage2 = cv2.resize(timeImage2, (28,28))/255.0
            y2_prediction = ocrmodel(tf.convert_to_tensor(np.expand_dims(timeImage2, axis=0), dtype=tf.float32))
            y2_prediction = y2_prediction.numpy().argmax()
            hms = hms+str(y2_prediction)
            splitImage2 = np.hstack((splitImage2, timeImage2))
    
        if i == 0:
            splitImage = splitImage2
            plt.figure(1, figsize=(10.0, 10.0))
            plt.axis('off')
            plt.imshow(img2)
            plt.pause(0.5)
            plt.figure(2, figsize=(4.0, 4.0))
            plt.axis('off')
            plt.imshow(splitImage2)
            print(hms)
        time = date+hms    
        
        # ------ YOLO detection for the boxes ------
        d = yolo.predict(frame)
        # len(d): num of objects
        # for d[i], dims 0 to 3 is position ranged [0, 1] -- center_x, center_y, w, h; dim 4 is class (0 for person); dim 5 is score.
        img_shape = frame.shape  # (480, 640, 3)
    
        if (all(d[:, 2]*d[:, 3] == 0) and i-ilast > fps/2) or cv2.waitKey(1) & 0xFF == ord(' '):
            frame_name = '{}_{}{}'.format(
                video_name_ext, i, desired_img_format)
            frame_path = os.path.join(folder_path, frame_name)
            cv2.imwrite(frame_path, frame)
            ilast = i
    
        # ------ get boxes ------
        bboxes = []
        for bbox in d:
            if (bbox[5] >= 0.4) and (bbox[4] == 0):
                # high score person
                for j in (1, 3):
                    bbox[j] *= img_shape[0]
                for j in (0, 2):
                    bbox[j] *= img_shape[1]
                # box position
                c_x = int(bbox[0])
                c_y = int(bbox[1])
                half_w = int(bbox[2] / 2)  # max(int(bbox[2] / 2),128)
                half_h = int(bbox[3] / 2)  # max(int(bbox[3] / 2),128)
                bboxes.append([bbox[5],c_x,c_y,half_w,half_h])
                # left = c_x - half_w
                # if left < 0:
                #     left = 0
                #     right = 2*half_w
                # top = c_y - half_h
                # if top < 0:
                #     top = 0
                #     bottom = 2*half_h
                # right = c_x + half_w
                # if right >= img_shape[1]:
                #     right = img_shape[1]-1
                #     left = right - 2*half_w
                # bottom = c_y + half_h
                # if bottom >= img_shape[0]:
                #     bottom = img_shape[0]-1
                #     top = bottom - 2*half_h
    
                # top_left = (left, top)
                # bottom_right = (right, bottom)
    
                # # ------ draw the box ------
                # cv2.rectangle(frame, top_left,
                #               bottom_right, (255, 0, 0), 1)
        if len(bboxes)>0:
            bboxes = np.array(bboxes)
            ind = np.argmax(bboxes[:,0])
            timeyolo.append([int(time),bboxes[ind,1],bboxes[ind,2],bboxes[ind,3],
                             bboxes[ind,4],lumi])
        else:
            timeyolo.append([int(time),0,0,0,0,lumi])
    
    
        
        # cv2.imshow("Demo", frame)
        # # videoWriter.write(frame)
        # if cv2.waitKey(1) & 0xFF == ord('\x1b'):
        #     break
    savemat(matpath, mdict = {'timeyolo': timeyolo})
    np.savetxt(txtpath,timeyolo,fmt='%d %d %d %d %d %d',newline='\n')
    
    # cv2.destroyAllWindows()
    cap.release()
    # videoWriter.release()
    fcnEnd = datetime.now()
    print(fcnEnd-fcnBegin)
    
    return fps, splitImage
