import numpy as np
import json
import re


def IOU(box1,box2):
    #assert box1.size()==4 and box2.size()==4,"bounding box coordinate size must be 4"
    bxmin = max(box1[0],box2[0])
    bymin = max(box1[1],box2[1])
    bxmax = min(box1[2],box2[2])
    bymax = min(box1[3],box2[3])
    bwidth = max(bxmax-bxmin,0)
    bhight = max(bymax-bymin,0)
    inter = bwidth*bhight
    union = (box1[2]-box1[0])*(box1[3]-box1[1])+(box2[2]-box2[0])*(box2[3]-box2[1])-inter
    
    return inter/union

def Dis(box1,box2):
    x1_mean = (box1[0]+box1[2])/2
    y1_mean = (box1[1]+box1[3])/2
    x2_mean = (box2[0]+box2[2])/2
    y2_mean = (box2[1]+box2[3])/2

    return abs(x1_mean-x2_mean)+abs(y1_mean-y2_mean)


file_path = "/mnt/sdf/shikra_output_rebuttal/hmba/test_extra_prediction.jsonl"


def str_to_float(str):
    str = str[1:-1]
    str = str.split(',')
    x1 = float(str[0])
    y1 = float(str[1])
    x2 = float(str[2])
    y2 = float(str[3])
    return [x1,y1,x2,y2]

with open(file_path, 'r') as file:
    idn = 0
    sum_iou = 0
    sum_dis =0
    acc = 0
    wrong = 0
    for line in file:
        data = json.loads(line)
        pred_withbox = data["pred"]
        target = data["target"]
        #print(pred_withbox)
        pred_cap = []
        pred_box = []
        st, ed = 0, 0
        for i in range(len(pred_withbox)):
            if pred_withbox[i] == '[':
                st = i
            if pred_withbox[i] == ']':
                ed = i
        pred_box = pred_withbox[st:ed+1]
        caption = pred_withbox[0:st] + pred_withbox[ed+1:]
        if len(pred_box)!=25:
            print(pred_box)
            print(pred_withbox)
            pass
        #print(pred_box)
        pred_box = str_to_float(pred_box)
        ##########处理target内容############
        st, ed = 0, 0
        l = len(target)
        for i in range(l):
            if target[l-1-i] == ']':
                ed = l-1-i
            if target[l-1-i] == '[':
                st = l-1-i
                break
        gt_box = target[st:ed+1]
        gt_box = str_to_float(gt_box)
        ##########计算IOU##################
        iou = IOU(pred_box, gt_box)
        dis = Dis(pred_box,gt_box)
        if iou == 0:
            wrong += 1
            #print(pred_withbox,target)
        if iou>0.5:
            acc += 1
        
        sum_iou += iou
        sum_dis += dis
        idn += 1
    print("mIOU", sum_iou/idn)
    print("Acc",  acc/idn)
    print("Wrong", wrong, wrong/idn)
    print("mDis", sum_dis/idn)
   
