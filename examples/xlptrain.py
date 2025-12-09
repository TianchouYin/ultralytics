import sys
sys.path.append("./")
from pathlib import Path

import yaml

import ultralytics
from ultralytics import YOLO
import os

from ultralytics.models.yolo import model


def xlptrain():
    ultralytics.checks()

    # model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

    # Load a model
    # model = YOLO('yolov8n.pt')
    # model = YOLO('D:/File/gitRepository/ultralytics-main/examples/runs/detect/train32/weights/best.pt')
    # model = YOLO('D:/File/gitRepository/ultralytics-main/examples/runs/detect/train7/weights/mcsgAloneBest180.pt')
    # mymodel = 'D:/File/gitRepository/ultralytics-main/examples/runs/detect/train4/weights/best.pt'  # epoch110 cuda out of memory
    # mymodel = 'D:/File/gitRepository/ultralytics-main/examples/runs/detect/train2McsgHalf2-Batch=4/weights/best.pt'
    # mymodel = 'runs/detect/train-halfv2AugV1-bc16_imgsz512_lr0_Mosaic1/weights/best.pt'

    # model = YOLO('yolov8x.yaml')
    model = YOLO('yolo12x.yaml')
    # model = YOLO('yolov8x-swingTransformer.yaml')
    # model = YOLO('yolov8x-C3STR_P3P5_P3.yaml')

    # model = YOLO('yolov8x_MaSA.yaml')
    # model = YOLO('MaSA/yolov8x_MaSA_P5.yaml')
    # model.load  ('yolov8x.pt')
    model.load  ('yolo12x.pt')
    # Load a model
    # model = YOLO("yolo12x.pt")
    # model.load  (r"D:\File\gitRepository\ultralytics-main\examples\runs\detect\trainyolov8x\train-yolov8xpng206-216+226-233Crp-Augv1++-1+-20_bc16imgsz320dpout0.5maxdet68pretrnFalse_pati10wks8deg1scl0.1lr0msc0_2\weights\last.pt")

    # p2： 从yolov8*.pt迁移学习，
    # model = YOLO('yolov8x-p2.yaml')
    # model.load('yolov8x.pt')

    # model = YOLO('yolov8x-addp2.yaml')
    # model.load('yolov8x.pt')

    # Train the model
    # results = model.train(data='coco8.yaml', epochs=1)
    # results = model.train(data='mcsg.yaml', epochs=300)
    # data = 'coco8.yaml'
    # data = 'D:/File/gitRepository/ultralytics-main/ultralytics/cfg/datasets/mcsgAlone.yaml'
    # data = 'D:/File/gitRepository/ultralytics-main/ultralytics/cfg/datasets/mcsgHalf.yaml'
    # data = 'D:/File/gitRepository/ultralytics-main/ultralytics/cfg/datasets/halfv2AugV1.yaml' #os
    # data = r'D:/File/gitRepository/ultralytics-main/ultralytics/cfg/datasets/mcsgHalfAugTrainSet-cropped.yaml'
    # data = r'D:/File/gitRepository/ultralytics-main/ultralytics/cfg/datasets/mcsgHalfAug-cropped.yaml'
    # data = r'D:/File/gitRepository/ultralytics-main/ultralytics/cfg/datasets/mcsgHalfAugv3-cropped.yaml'
    # data = r'D:/File/gitRepository/ultralytics-main/ultralytics/cfg/datasets/mcsgHalfAugv4-cropped.yaml'
    # data = r'D:/File/gitRepository/ultralytics-main/ultralytics/cfg/datasets/mcsgHalfAug+-1-cropped.yaml'

    #中华医学杂志
    # data = r'D:\File\gitRepository\yolov9-main\data\png206-216+226-233Crp-Augv1++-1+-20.yaml'
    # data = r"D:\File\gitRepository\yolov9-main\data\png206-216+226-233Crp.yaml"

    workDir = Path(os.getcwd()).parent
    # data = str(workDir.parent) +r"/ultralytics-main/ultralytics/cfg/datasets/css50png.yaml"
    data = r"./ultralytics/cfg/datasets/CASnoRadom.yaml"


    # model.train(data='D:/File/gitRepository/ultralytics-main/ultralytics/cfg/datasets/mcsgHalfv2Gray16.yaml', batch=4, name='trainHalfv2Gray16-batch4' )
    mymodelName,_ = os.path.splitext(model.cfg) # yolov8-p2x.yaml

    # mymodelName = model.ckpt['train_args']['model']   # 基础权重的文件名

    # 从YAML文件中加载字典
    with open('/data/users/lxing/gitRepository/ultralytics250316/ultralytics/cfg/default.yaml', 'r',encoding='utf-8') as file:
        yaml_dict = yaml.safe_load(file)
    xlpMaxdet = 'maxdet'+str(yaml_dict['max_det'])
    xlpPretrain = 'pretrn'+str(yaml_dict['pretrained'])+'_'
    xlppatience = 'pati'+str(yaml_dict['patience'])
    xlpWorkers = 'wks'+str(yaml_dict['workers'])
    bc = 'bc'+str(yaml_dict['batch'])
    myimgsz = 'imgsz'+str(yaml_dict['imgsz'])
    dropout = 'dpout'+str(yaml_dict['dropout'])
    dataset = os.path.basename(data).strip('.yaml')
    clsloss = 'cls'+str(yaml_dict['cls'])
    boxloss = 'box'+str(yaml_dict['box'])
    dfl = 'dfl'+str(yaml_dict['dfl'])
    singlecls = 'singcls'+str(yaml_dict['single_cls'])
    if yaml_dict['single_cls']== False:
        singlecls = ''

    # xlpOutName = 'train-'+mymodelName+'_halfv2AugV1-bc'+str(bc)+'imgsz'+str(myimgsz)+'dpout0.5'+xlpMaxdet+xlpPretrain+xlppatience+xlpWorkers+'_'
    # xlpOutName = (mymodelName+'\\'+'train-'+mymodelName+'c3strTrue-'+dataset+'_'+bc+myimgsz+dropout+xlpMaxdet+xlpPretrain+xlppatience+xlpWorkers+'deg'+str(yaml_dict['degrees'])+
    #               'scl'+str(yaml_dict['scale'])+'lr'+str(yaml_dict['fliplr'])+'msc'+str(yaml_dict['mosaic'])+'_')
    # xlpOutName = ('C3STR-'+dataset+'\\'+mymodelName+'\\'+'train'+'\\'+bc+myimgsz+dropout)
    xlpOutName = (dataset+'\\'+mymodelName+'\\'+'train'+'\\'+bc+myimgsz+dropout)
    # xlpOutName = (dataset+'\\'+mymodelName+'\\'+'train'+'\\'+bc+myimgsz+dropout+'Loss'+clsloss+boxloss+dfl+singlecls+'(')
    # model.train(data='D:/File/gitRepository/ultralytics-main/ultralytics/cfg/datasets/halfv2AugV1.yaml', batch=bc, imgsz=myimgsz, name='train-halfv2AugV1-bc'+str(bc)+'_imgsz'+str(myimgsz)+'_lr0_Mosaic1' )
    model.train(data=data, name=xlpOutName,cfg='/data/users/lxing/gitRepository/ultralytics250316/ultralytics/cfg/default.yaml')

if __name__ ==  '__main__':
    xlptrain()

