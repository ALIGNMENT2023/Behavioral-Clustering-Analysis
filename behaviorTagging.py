from datetime import datetime,timedelta
import os
os.environ['CUDA_VISIBLE_DEVICES']='3'
import tensorflow as tf

import cv2
import numpy as np
from scipy.io import savemat
from tqdm import tqdm

blank_size = 0
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.6
  
def nonblank_lines(f):
    for l in f:
        line = l.rstrip()
        if line:
            yield line
            
# load class list
with open('D:/data/git/ActionLabeling/class_list_test.txt') as f:
# with open('class_list_test.txt') as f:
    CLASS_LIST = list(nonblank_lines(f))
class_num = len(CLASS_LIST)   
   
model = tf.keras.applications.ResNet50(weights=None,input_shape=(576,1024,3),classes=class_num)
time_pre_train = '' # user define, like'20210703-191449'#'20210502-223408'#'20210409-103305'
best_pre_train = 121#350#120
checkpoint_path = "D:/data/git/ActionLabeling/checkpoints/{time:s}/cp-{epoch:03d}.ckpt"
model.load_weights(checkpoint_path.format(time=time_pre_train, epoch=best_pre_train))


def behaviorTagging(filePath='S08-20210119-12', camType=4):          
    # cap = cv2.VideoCapture(0)
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
    matpath = os.path.join(folderpath, 'frameClass'+str(camType)+'-new.mat')
    # srtpath = os.path.join(folderpath, 'frameClass'+str(camType)+'.srt')
    # if os.path.exists(srtpath):
    #     os.remove(srtpath)
    if os.path.exists(matpath):
        return
    
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
    # srtname = os.path.splitext(video2read)[0]+'.srt'
    # srtpath = os.path.join(folderpath,srtname)
    srtpath = os.path.splitext(vidpath)[0]+'.new.srt'
    # vidpath='D:/data/AnalysisPKU/S08/20200520/AS08-20200520-02/YiVS08-20200520-02-L02l.mp4'
    # vid2write='D:/data/AnalysisPKU/S08/20200520/AS08-20200520-02/YiVS08-20200520-02-L02l-resnet3.mp4'
    # matpath = 'D:/data/AnalysisPKU/S08/20200520/AS08-20200520-02/YiVS08-20200520-02-L02l-frameClass.mat'
    cap = cv2.VideoCapture(vidpath)
    # folder_path = 'D:/data/Pictures/failframe/yolov4'
    # file_path, file_extension = os.path.splitext(vidpath)
    # file_extension = file_extension.replace('.', '_')
    # file_path += file_extension
    # video_name_ext = os.path.basename(file_path)
    # desired_img_format = '.jpg'
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # videoWriter = cv2.VideoWriter(vid2write, cv2.VideoWriter_fourcc('M','P','E','G'), fps, (int(width),int(height)))
    # ilast = -1
    tbegin = datetime.now()
    frame_class_name = []
    frame_class_index = []
    frame_class_value = []
    for i in tqdm(range(n_frames)):
    # while(1):
        ret, frame = cap.read()
        if not ret:
            break
        img = frame.copy()
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (1024,576))
        img = tf.convert_to_tensor(np.expand_dims(img, axis=0), dtype=tf.float32)
        y = model(img)
        ind = np.argmax(y[0].numpy())
        class_name = CLASS_LIST[ind]
        frame_class_name.append(class_name)
        frame_class_index.append(ind)
        frame_class_value.append(y)
        # cv2.putText(frame, class_name, (int(0.8*width), int(0.2*height)), font, font_scale, (0, 0, 255), 1, cv2.LINE_AA)
                
        # cv2.imshow("test", frame)
        # videoWriter.write(frame)
        # if cv2.waitKey(1) & 0xFF == ord('\x1b'):
            # break
    # cv2.destroyAllWindows()
    cap.release()
    # videoWriter.release()
    frame_class_index=np.array(frame_class_index)
    # di = np.diff(frame_class_index)
    # di = np.nonzero(di)[0]
    iLeft = np.nonzero(np.diff(np.concatenate(([-1],frame_class_index))))[0]
    iRight = np.nonzero(np.diff(np.append(frame_class_index,-1)))[0]
    with open(srtpath,'w') as f:
        start_time=datetime.strptime('00:00:00','%H:%M:%S')
        for i in range(len(iLeft)):
            f.write('{}\n'.format(i))
            t1 = timedelta(seconds=iLeft[i]/fps)
            t2 = timedelta(seconds=iRight[i]/fps)
            t1 = datetime.strftime(t1+start_time,'%H:%M:%S,%f')
            t1 = t1[:-3]
            t2 = datetime.strftime(t2+start_time,'%H:%M:%S,%f')
            t2 = t2[:-3]
            f.write(t1+' --> '+t2+'\n')
            f.write(frame_class_name[iLeft[i]]+'\n'+'\n')
            
    frame_class_name=np.array(frame_class_name)
    frame_class_value=np.squeeze(np.array(frame_class_value))
    savemat(matpath,{'class_index':frame_class_index,'class_name':frame_class_name,
                     'class_value':frame_class_value,'fps':fps,'tagName':CLASS_LIST})
    tend = datetime.now()
    telapse = (tend-tbegin).seconds
    print('time elpase is {} seconds'.format(telapse))

