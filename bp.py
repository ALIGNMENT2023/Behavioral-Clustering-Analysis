#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 17:53:59 2017

@author: yangx
"""


def batchBatch(analysisMode):
    import os
    import glob
#    import tkFileDialog
    import tkinter as tk
    from tkinter import filedialog
#    import time

#    defaultFolder = '/media/yangx/YXHD-01/data_shared/mouseTouchData4Analysis/'
#    defaultFolder = 'C:\\Users\\Admin\\Documents\\mouseTouchData'
    defaultFolder = 'D:/Data/mouseTouchData4Analysis'
    currentpath = os.getcwd()
    os.chdir(defaultFolder)
    root = tk.Tk()
#    foldersPath = tkFileDialog.askdirectory()
    foldersPath = filedialog.askdirectory()
    folderList = glob.glob(os.path.join(foldersPath, '*'))
    folderList.sort()
#    time.sleep(4200)
    root.destroy()
    for folderPath in folderList:
        if os.path.isdir(folderPath):
            batchProcessingPKU(analysisMode, folderPath)
    os.chdir(currentpath)
    return(0)


def batchProcessingPKU(analysisMode, folderPath=None, parallel=False, type='top'):
    import os
    import glob
#    import tkFileDialog
    import tkinter as tk
    from tkinter import filedialog
    from joblib import delayed
    from joblib import Parallel
    import time
    from mouseTouch import vs
    from mouseTouch import tr
    from mouseTouch import ta
    from mouseTouch import va
#    from vs import videoStitchingPKU
#    from tr import timeReadingPKU
#    from ta import timelineAligningPKU
#    from ta import correctTSmanuallyPKU
#    from va import videoClipPKU
#    import pdb
#    pdb.set_trace()
    fcnBegin = time.time()
#    defaultFolder = '/media/yangx/YXHD-01/data_shared/mouseTouchData4Analysis/'
#    defaultFolder = 'C:\\Users\\Admin\\Documents\\mouseTouchData'
    defaultFolder = 'D:/Data/mouseTouchData4Analysis'
    currentpath = os.getcwd()
    os.chdir(defaultFolder)
    if folderPath is None:
        root = tk.Tk()
        folderPath = filedialog.askdirectory()
#        folderPath = tkFileDialog.askdirectory()
        root.destroy()
    print(folderPath)
    txtList = glob.glob(os.path.join(folderPath, '*.txt'))
    txtList.sort()
#    time.sleep(4200)
    if type=='top':
        camNum = 3
    if type=='side':
        camNum = 4
    if parallel:
        if analysisMode & 1 == 1:
            status_lst = Parallel(n_jobs=5)(delayed(vs.videoStitchingPKU)(
                filename.replace('\\', '/'), camNum) for filename in txtList)
        if analysisMode & 4 == 4:
            status_lst = Parallel(n_jobs=5)(delayed(ta.timelineAligningPKU)(
                filename.replace('\\', '/'), camType=camNum) for filename in txtList)
        if analysisMode & 8 == 8:
            status_lst = Parallel(n_jobs=5)(delayed(va.videoClipPKU)(
                filename.replace('\\', '/'), 70, camNum) for filename in txtList)
        if analysisMode & 64 == 64:
            status_lst = Parallel(n_jobs=5)(delayed(va.vidCalRes2)(
                filename.replace('\\', '/')) for filename in txtList)
        if analysisMode & 128 == 128:
            status_lst = Parallel(n_jobs=5)(delayed(ta.alignByLED)(
                filename.replace('\\', '/'), camType=camNum) for filename in txtList)
        if analysisMode & 256 == 256:
            status_lst = Parallel(n_jobs=5)(delayed(vs.videoCompressingPKU)(
                filename.replace('\\', '/'), camNum) for filename in txtList)
        print(status_lst)
    else:
        if analysisMode & 1 == 1:
            root = tk.Tk()
            tfPath = filedialog.askdirectory()
            root.destroy()
        for filename in txtList:
            filename = filename.replace('\\', '/')
            print(filename)
            if analysisMode & 1 == 1:
                vs.videoStitchingOnlyPKU(filename, camNum, tfPath)
            if analysisMode & 2 == 2:
                tr.timeReadingPKU3(filename, camType=camNum)
            if analysisMode & 4 == 4:
                ta.timelineAligningPKU(filename, camType=camNum)
            if analysisMode & 8 == 8:
                va.videoClipPKU(filename, 70, camNum)
            if analysisMode & 16 == 16:
                # timeError=(timeStampCorrected-timeStamp)-(Origin-L)
                ta.correctTSmanuallyPKU(filename, -23, camNum)
            if analysisMode & 32 == 32:
                va.delVideoPKU(filename, camNum)
            if analysisMode & 64 == 64:
                va.vidCalRes2(filename)
            if analysisMode & 128 == 128:
                ta.alignByLED(filename, camType=camNum)
            if analysisMode & 256 == 256:
                #                vs.videoCompressingPKU(filename, camNum)
                vs.videoRenamePKU(filename, 2)
#                vs.deleteVideoPKU(filename, camNum)
        if analysisMode & 2 == 2:
            restart_kernel()

    os.chdir(currentpath)
    fcnEnd = time.time()
    telapse = (fcnEnd-fcnBegin)/60
    print('time elapse is %.1f min\n' % telapse)
    return(0)


def batchFreezingPKU():
    import os
    import glob
#    import tkFileDialog
    import tkinter as tk
    from tkinter import filedialog
    import numpy as np
#    from va import freezingDetectionPKU
#    from va import arenaMaskPKU
    from mouseTouch import va
#    import mouseTouch
#    defaultFolder = '/media/yangx/YXHD-01/data_shared/mouseTouchData4Analysis/'
#    defaultFolder = 'C:\\Users\\Admin\\Documents\\mouseTouchData'
    defaultFolder = 'D:/Data/mouseTouchData4Analysis'
    currentpath = os.getcwd()
    os.chdir(defaultFolder)
    root = tk.Tk()
#    folderPath = tkFileDialog.askdirectory()
    folderPath = filedialog.askdirectory()
    txtList = glob.glob(os.path.join(folderPath, '*.txt'))
    txtList.sort()
    root.destroy()
#    time.sleep(4200)

#    analysisfolder = '/media/yangx/YXHD-01/data_shared/AnalysisPKU'
#    analysisfolder = 'C:\\Users\\Admin\\Documents\\Analysis'
    analysisfolder = 'D:/Data/AnalysisPKU'
#    analysisSetupFolder = os.path.join(analysisfolder,setupFolder)
    prefix = 'YiV'
    begin = True

    for filePath in txtList:
        if begin:
            va.arenaMaskPKU(filePath)
            begin = False
#        parentfolder = os.path.split(filePath)[0]
#        parentfolder = os.path.split(parentfolder)[1]
        filename = os.path.split(filePath)[1]
        filename = os.path.splitext(filename)[0]
        if filename[0] == 'S' or filename[0] == 's':
            setupFolder = filename[0:3]
            parentfolder = filename[4:12]
        else:
            setupFolder = 'siat'
            parentfolder = filename[0:8]
        subfolder = 'A'+filename
        folderpath = os.path.join(
            analysisfolder, setupFolder, parentfolder, subfolder)
        video2read = prefix+filename+'-L??.mp4'
        mp4List = glob.glob(os.path.join(folderpath, video2read))
        for iloop in range(np.size(mp4List)):
            va.freezingDetectionPKU(filePath, iloop+1)
    os.chdir(currentpath)
    return(0)


def batchBatchAnalysis(analysisMode):
    import os
    import glob
#    import tkFileDialog
    import tkinter as tk
    from tkinter import filedialog
#    defaultFolder = '/media/yangx/YXHD-01/data_shared/mouseTouchData4Analysis/'
#    defaultFolder = 'C:\\Users\\Admin\\Documents\\mouseTouchData'
    defaultFolder = '' # user defines, like 'D:/Data/mouseTouchData4Analysis'
    currentpath = os.getcwd()
    os.chdir(defaultFolder)
    root = tk.Tk()
#    foldersPath = tkFileDialog.askdirectory()
    foldersPath = filedialog.askdirectory()
    folderList = glob.glob(os.path.join(foldersPath, '*'))
    folderList.sort()
#    time.sleep(4200)
    root.destroy()
    for folderPath in folderList:
        if os.path.isdir(folderPath):
            batchAnalysisPKU(analysisMode, folderPath)
    os.chdir(currentpath)
    return(0)


def batchAnalysisPKU(analysisMode, folderPath=None, parallel=False):
    import os
    import glob
#    import tkFileDialog
    import tkinter as tk
    from tkinter import filedialog
    import numpy as np
    from joblib import delayed, Parallel
#    from va import freezingDetectionPKU
#    from va import behaviorAnalysis
#    from va import arenaMaskPKU
    from mouseTouch import va

    
    if analysisMode & 4 == 4:
        from mouseTouch import mouseDetection as md
    elif analysisMode & 8 == 8:     
        from mouseTouch import recordAlignment as ra
    elif analysisMode & 16 == 16:
        from mouseTouch import behaviorTagging as bt
    elif analysisMode & 32 == 32:
        from mouseTouch import randomForest4unclear as rf
    elif analysisMode & 64 == 64:
        from mouseTouch import tagCluster as tc
    elif analysisMode & 128 == 128:    
        from mouseTouch import caption as st
    import time

    fcnBegin = time.time()
#    defaultFolder = '/media/yangx/YXHD-01/data_shared/mouseTouchData4Analysis/'
    defaultFolder = '' # user defines, like 'D:/data/mouseTouchData4Analysis/data4stat/'
    currentpath = os.getcwd()
    os.chdir(defaultFolder)
    if folderPath is None:
        root = tk.Tk()
        folderPath = filedialog.askdirectory()
        root.destroy()
    print(folderPath)
    txtList = glob.glob(os.path.join(folderPath, '*.txt'))
    txtList.sort()
#    time.sleep(4200)

#    analysisfolder = '/media/yangx/YXHD-01/data_shared/AnalysisPKU'
#    analysisfolder = 'D:/data/Data/AnalysisPKU'
#    analysisSetupFolder = os.path.join(analysisfolder,setupFolder)
#    prefix = 'YiV'
#    begin = True
    if analysisMode & 1 == 1:
        va.arenaMaskPKU(txtList[0].replace('\\', '/'))
    if parallel:
        if analysisMode & 1 == 1:
            status_lst = Parallel(n_jobs=5)(delayed(va.freezingDetectionPKU)(
                filename.replace('\\', '/')) for filename in txtList)
        if analysisMode & 2 == 2:
            status_lst = Parallel(n_jobs=5)(delayed(va.behaviorAnalysis)(filename.replace(
                '\\', '/'), wholeVid=True) for filename in txtList)
        if analysisMode & 4 == 4:
            status_lst = Parallel(n_jobs=3)(delayed(md.mouseDetection)(filename.replace(
                '\\', '/'), np.mod(i, 3)+1, 3) for i, filename in enumerate(txtList))
        if analysisMode & 8 == 8:
            status_lst = Parallel(n_jobs=5)(delayed(ra.recordAlignment)(filename.replace(
                '\\', '/'), camType=4) for filename in txtList)
        if analysisMode & 16 == 16:
            status_lst = Parallel(n_jobs=5)(delayed(bt.behaviorTagging)(filename.replace(
                '\\', '/'), camType=4) for filename in txtList)

    else:
        for filePath in txtList:
            print(filePath)
            fps = 30
            if analysisMode & 1 == 1:
                va.freezingDetectionPKU(filePath.replace('\\', '/'))
            if analysisMode & 2 == 2:
                va.behaviorAnalysis(filePath.replace('\\', '/'))
            if analysisMode & 4 == 4:
                md.mouseDetection(
                    filePath.replace('\\', '/'), camType = 3)
            if analysisMode & 8 == 8:
                ra.recordAlignment(filePath.replace(
                    '\\', '/'), camType = 4)
            if analysisMode & 16 == 16:
                bt.behaviorTagging(filePath.replace(
                    '\\', '/'), camType = 4)    
            if analysisMode & 32 == 32:
                rf.randomForest4unclearTag(filePath.replace(
                    '\\', '/'))
                # rf.tagBySavedModel(filePath.replace(
                #     '\\', '/'))
                # ra.frameSelectionbyLocation(filePath.replace(
                #     '\\', '/'), camType=3)
                # ra.frameSelectionbyLocation(filePath.replace(
                #     '\\', '/'), camType=4)
            if analysisMode & 64 == 64:
                # tc.tagCluster(filePath.replace(
                #     '\\', '/'))
                tc.clusterByModel(filePath.replace(
                    '\\', '/'))
            if analysisMode & 128 == 128:
                st.srt4vid(filePath.replace(
                    '\\', '/'))
#        if begin and analysisMode&1==1:
#            va.arenaMaskPKU(filePath)
#            begin = False
#        parentfolder = os.path.split(filePath)[0]
#        parentfolder = os.path.split(parentfolder)[1]

#            filename = os.path.split(filePath)[1]
#            filename = os.path.splitext(filename)[0]
#            if filename[0]=='S' or filename[0]=='s':
#                setupFolder = filename[0:3]
#                parentfolder = filename[4:12]
#            else:
#                setupFolder = 'siat'
#                parentfolder = filename[0:8]
#            subfolder = 'A'+filename
#            folderpath = os.path.join(analysisfolder,setupFolder,parentfolder,subfolder)
#            video2read = prefix+filename+'-L??.mp4'
#            mp4List = glob.glob(os.path.join(folderpath,video2read))
#            for iloop in range(np.size(mp4List)):
#                if analysisMode&1==1:
#                    va.freezingDetectionPKU(filePath,iloop+1)
#                if analysisMode&2==2:
#                    va.behaviorAnalysis(filePath,iloop+1)

    os.chdir(currentpath)
    fcnEnd = time.time()
    telapse = (fcnEnd-fcnBegin)/60
    print('\nTime elapse is %.1f min\n' % telapse)
    return(0)


def restart_kernel():
    import time
    import sys
    import os

    print('Python kernel will restart after 3 seconds.')
    time.sleep(3)
    python = sys.executable
    os.execl(python, python, *sys.argv)
