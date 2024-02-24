import os
from heart_utils import *
from utils import *
import torch
import numpy as np
import nibabel
from albumentations.pytorch.transforms import ToTensorV2
import sys
from albumentations import *
from models.classification_3d import Resnet50_3d
from models.segmentation_2d import Segnet
import statistics
import argparse

def process_images(ed_path,es_path,device):

    channels  = [16, 32, 64, 128] #no of layers of the segnet model architecture

    model_fold1 = Segnet(num_features=1, num_outputs=4, channels=[16, 32, 64, 128], num_residual=2).to(device)
    model_fold1_parameters = filter(lambda p: p.requires_grad, model_fold1.parameters())
    model_fold1.load_state_dict(torch.load('model_weights/segnet_fold1.pth', map_location=device)) #load the model weights for fold1
    model_fold1.eval()

    model_fold2 = Segnet(num_features=1, num_outputs=4, channels=[16, 32, 64, 128], num_residual=2).to(device)
    model_fold2_parameters = filter(lambda p: p.requires_grad, model_fold2.parameters())
    model_fold2.load_state_dict(torch.load('model_weights/segnet_fold2.pth', map_location=device)) #load the model weights for fold2
    model_fold2.eval()

    model_fold3 = Segnet(num_features=1, num_outputs=4, channels=[16, 32, 64, 128], num_residual=2).to(device)
    model_fold3_parameters = filter(lambda p: p.requires_grad, model_fold3.parameters())
    model_fold3.load_state_dict(torch.load('model_weights/segnet_fold3.pth', map_location=device)) #load the model weights for fold3
    model_fold3.eval()

    model_fold4 = Segnet(num_features=1, num_outputs=4, channels=[16, 32, 64, 128], num_residual=2).to(device)
    model_fold4_parameters = filter(lambda p: p.requires_grad, model_fold4.parameters())
    model_fold4.load_state_dict(torch.load('model_weights/segnet_fold4.pth', map_location=device)) #load the model weights for fold4
    model_fold4.eval()

    model_fold5 = Segnet(num_features=1, num_outputs=4, channels=[16, 32, 64, 128], num_residual=2).to(device)
    model_fold5_parameters = filter(lambda p: p.requires_grad, model_fold5.parameters())
    model_fold5.load_state_dict(torch.load('model_weights/segnet_fold5.pth', map_location=device)) #load the model weights for fold5
    model_fold5.eval()

    model_fold1_classification = Resnet50_3d().to(device)  #classification model architecture                         
    model_fold1_classification.load_state_dict(torch.load('model_weights/diagnosis_fold1.pth', map_location=device)) #load the model weights for fold1
    model_fold1_classification.eval()

    model_fold2_classification = Resnet50_3d().to(device)                          
    model_fold2_classification.load_state_dict(torch.load('model_weights/diagnosis_fold2.pth', map_location=device)) #load the model weights for fold2
    model_fold2_classification.eval()

    model_fold3_classification = Resnet50_3d().to(device)                          
    model_fold3_classification.load_state_dict(torch.load('model_weights/diagnosis_fold3.pth', map_location=device)) #load the model weights for fold3
    model_fold3_classification.eval()

    model_fold4_classification = Resnet50_3d().to(device)                          
    model_fold4_classification.load_state_dict(torch.load('model_weights/diagnosis_fold4.pth', map_location=device)) #load the model weights for fold4
    model_fold4_classification.eval()

    model_fold5_classification = Resnet50_3d().to(device)                          
    model_fold5_classification.load_state_dict(torch.load('model_weights/diagnosis_fold5.pth', map_location=device)) #load the model weights for fold5
    model_fold5_classification.eval()

    train_feat_means_fold_1 = np.load('weights/train_feat_means_fold_1.npy') #load the features weights of fold1
    train_feat_stds_fold_1 = np.load('weights/train_feat_stds_fold_1.npy') #load the features weights of fold1

    train_feat_means_fold_2 = np.load('weights/train_feat_means_fold_2.npy') #load the features weights of fold2
    train_feat_stds_fold_2 = np.load('weights/train_feat_stds_fold_2.npy') #load the features weights of fold2

    train_feat_means_fold_3 = np.load('weights/train_feat_means_fold_3.npy') #load the features weights of fold3
    train_feat_stds_fold_3 = np.load('weights/train_feat_stds_fold_3.npy') #load the features weights of fold3

    train_feat_means_fold_4 = np.load('weights/train_feat_means_fold_4.npy') #load the features weights of fold4
    train_feat_stds_fold_4 = np.load('weights/train_feat_stds_fold_4.npy') #load the features weights of fold4

    train_feat_means_fold_5 = np.load('weights/train_feat_means_fold_5.npy') #load the features weights of fold5
    train_feat_stds_fold_5 = np.load('weights/train_feat_stds_fold_5.npy') #load the features weights of fold5

    labels = ['HCM', 'DCM', 'NOR', 'MINF', 'RV'] #labels

    ed = 0
    es = 0 
    ed_spaing = 0
    es_spacing = 0
    ed_path_new = ''
    es_path_new = ''
    for_which_classes = [(3), ]
    min_valid_object_size:dict=None

    Resize = augmentations.transforms.Resize(256, 256)
    val_both_aug = Compose([Resize])

    for idx,path in enumerate([ed_path,es_path]): #iterate through the images
        
        im1 = nibabel.load(path) #Load the image
        volume_per_voxel = np.prod(im1.header.get_zooms(), dtype=np.float64)
        img = im1.get_fdata().astype('float32')
        img = img/img.max() #normalize the image
        new_masks = []
        images = []

        for i in range(img.shape[2]): #Iterate through the z-axis

            img1 = img[...,i][...,None] 
            augmented = val_both_aug(image=img1, mask=img1) #Augment the image as used for validation
            img1 = augmented['image']
            to_tensor = ToTensorV2()
            img1 = np.clip(img1, 0, 1)
            img1 = (img1-0.5)/0.5 
            to_tensor = ToTensorV2()
            img1 = to_tensor(image=img1)
            img1 = img1['image']
            #Model Inference
            y_pred_fold1 = model_fold1(img1[None,...].to(device)).detach() #Use fold1 prediction
            y_pred_fold1 = y_pred_fold1[0,...]

            y_pred_fold2 = model_fold2(img1[None,...].to(device)).detach() #Use fold2 prediction
            y_pred_fold2 = y_pred_fold2[0,...] 

            y_pred_fold3 = model_fold3(img1[None,...].to(device)).detach() #Use fold3 prediction
            y_pred_fold3 = y_pred_fold3[0,...]

            y_pred_fold4 = model_fold4(img1[None,...].to(device)).detach() #Use fold4 prediction
            y_pred_fold4 = y_pred_fold4[0,...]

            y_pred_fold5 = model_fold5(img1[None,...].to(device)).detach() #Use fold5 prediction
            y_pred_fold5 = y_pred_fold5[0,...]

            y_pred = y_pred_fold1+y_pred_fold2+y_pred_fold3+y_pred_fold4+y_pred_fold5 #Get all the predictions of all folds

            predicted_mask = np.argmax(y_pred.detach().cpu(), 0).numpy() #Get the max prediction accross each fold
            predicted_mask = predicted_mask.astype('float32')
            new_mask = cv2.resize(predicted_mask,(img.shape[1],img.shape[0]),interpolation = cv2.INTER_NEAREST)
            new_img = cv2.resize(img1[0].cpu().numpy(),(img.shape[1],img.shape[0]),interpolation = cv2.INTER_NEAREST) #Resize the mask prediction to the original image size
            images.append(new_img)
            new_masks.append(new_mask)

        mask_new = np.array(new_masks)
        mask_new = np.transpose(mask_new,(1,2,0))
        mask_new,_,_ = load_remove_save(mask_new, volume_per_voxel ,for_which_classes, min_valid_object_size) #Postprocess the mask
        
        if idx ==0:
            ed = mask_new
            ed_spacing = im1.header.get_zooms()
            ed_path_new = path
            frame1_img = nibabel.load(ed_path_new).get_fdata()
        else:
            es = mask_new
            es_spacing = im1.header.get_zooms()
            es_path_new = path
            frame2_img = nibabel.load(es_path_new).get_fdata()
    
    ed_lv, ed_rv, ed_myo = heart_metrics(ed,
                ed_spacing) #Use the diastole mask file to get the quantitative stats
    es_lv, es_rv, es_myo = heart_metrics(es,
                    es_spacing) #Use the systole mask file to get the quantitative stats
    ef_lv = ejection_fraction(ed_lv, es_lv)
    ef_rv = ejection_fraction(ed_rv, es_rv)
    
    #calculate the stats using the mask files
    
    myo_properties = myocardial_thickness(es,es_spacing)
    es_myo_thickness_max_avg = np.amax(myo_properties[0])
    es_myo_thickness_std_avg = np.std(myo_properties[0])
    es_myo_thickness_mean_std = np.mean(myo_properties[1])
    es_myo_thickness_std_std = np.std(myo_properties[1])

    myo_properties = myocardial_thickness(ed,ed_spacing)
    ed_myo_thickness_max_avg = np.amax(myo_properties[0])
    ed_myo_thickness_std_avg = np.std(myo_properties[0])
    ed_myo_thickness_mean_std = np.mean(myo_properties[1])
    ed_myo_thickness_std_std = np.std(myo_properties[1])



    #append all the stats in the dictionary
    heart_param = {'EDV_LV': ed_lv, 'EDV_RV': ed_rv, 'ESV_LV': es_lv, 'ESV_RV': es_rv,
        'ED_MYO': ed_myo, 'ES_MYO': es_myo, 'EF_LV': ef_lv, 'EF_RV': ef_rv,
        'ES_MYO_MAX_AVG_T': es_myo_thickness_max_avg, 'ES_MYO_STD_AVG_T': es_myo_thickness_std_avg, 'ES_MYO_AVG_STD_T': es_myo_thickness_mean_std, 'ES_MYO_STD_STD_T': es_myo_thickness_std_std,
        'ED_MYO_MAX_AVG_T': ed_myo_thickness_max_avg, 'ED_MYO_STD_AVG_T': ed_myo_thickness_std_avg, 'ED_MYO_AVG_STD_T': ed_myo_thickness_mean_std, 'ED_MYO_STD_STD_T': ed_myo_thickness_std_std,}
    r=[]

    r.append(heart_param['EDV_LV'])
    r.append(heart_param['ESV_LV'])
    r.append(heart_param['EDV_RV'])
    r.append(heart_param['ESV_RV'])
    r.append(heart_param['ED_MYO'])
    r.append(heart_param['ES_MYO'])
    r.append(heart_param['EF_LV'])
    r.append(heart_param['EF_RV'])
    r.append(ed_lv/ed_rv)
    r.append(es_lv/es_rv)
    r.append(ed_myo/ed_lv)
    r.append(es_myo/es_lv)
    # r.append(patient_data[pid]['Height'])
    # r.append(patient_data[pid]['Weight'])
    r.append(heart_param['ES_MYO_MAX_AVG_T'])
    r.append(heart_param['ES_MYO_STD_AVG_T'])
    r.append(heart_param['ES_MYO_AVG_STD_T'])
    r.append(heart_param['ES_MYO_STD_STD_T'])

    r.append(heart_param['ED_MYO_MAX_AVG_T'])
    r.append(heart_param['ED_MYO_STD_AVG_T'])
    r.append(heart_param['ED_MYO_AVG_STD_T'])
    r.append(heart_param['ED_MYO_STD_STD_T'])
    
    
    #combine both the image and mask files in a single input array
    img = np.concatenate([
    frame1_img[None, ...], 
    frame2_img[None, ...],
    ed[None, ...],
    es[None, ...]
    ]).astype('float32')

    img[:2, ...] = img[:2, ...] / 1000 # scale image
    crop = CenterCrop_new(size=(312, 312, 16))
    img = crop(img)
    feats = np.array(r[:21]).astype('float32')
    img = torch.from_numpy(img).to(device)
    
    #normalize the quantitative stats obtained accross each fold
    
    feats1 = (feats - train_feat_means_fold_1) / train_feat_stds_fold_1
    feats1 = torch.from_numpy(feats1).to(device).float()

    feats2 = (feats - train_feat_means_fold_2) / train_feat_stds_fold_2
    feats2 = torch.from_numpy(feats2).to(device).float()

    feats3 = (feats - train_feat_means_fold_3) / train_feat_stds_fold_3
    feats3 = torch.from_numpy(feats3).to(device).float()

    feats4 = (feats - train_feat_means_fold_4) / train_feat_stds_fold_4
    feats4 = torch.from_numpy(feats4).to(device).float()

    feats5 = (feats - train_feat_means_fold_5) / train_feat_stds_fold_5
    feats5 = torch.from_numpy(feats5).to(device).float()

    img = img[None, ...]
    feats1 = feats1[None, ...]
    feats2 = feats2[None, ...]
    feats3 = feats3[None, ...]
    feats4 = feats4[None, ...]
    feats5 = feats5[None, ...]
    
    #pass the image and also the features to the classification model
    with torch.no_grad():
        prediction_fold_1 = model_fold1_classification(img, feats1)
    with torch.no_grad():
        prediction_fold_2 = model_fold2_classification(img, feats2)
    with torch.no_grad():
        prediction_fold_3 = model_fold3_classification(img, feats3)
    with torch.no_grad():
        prediction_fold_4 = model_fold4_classification(img, feats4)
    with torch.no_grad():
        prediction_fold_5 = model_fold5_classification(img, feats5)
    
    #Take the highest number of votes for each class across all the folds to get final prediction
    final_prediction = statistics.mode([prediction_fold_1.argmax(1).item(),prediction_fold_2.argmax(1).item(),prediction_fold_3.argmax(1).item(),\
                          prediction_fold_4.argmax(1).item(),prediction_fold_5.argmax(1).item()])

    print(f'Patient belongs to:{labels[final_prediction]} class')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ACDC diagnosis')  
    parser.add_argument('--ed_path', type=str, default='na', help='Diastole mri image')
    parser.add_argument('--es_path', type=str, default='na', help='Systole mri image')
    parser.add_argument('--device', type=str, default='cpu', help='Device name on which to infer the model')
    args = parser.parse_args()

    ed_path = args.ed_path
    es_path = args.es_path
    device = args.device

    print('\nProcessing images.....\n')
    process_images(ed_path,es_path,device)
    print('\nFinished running script\n')