import yaml
import os
from models.segmentation_2d import Segnet
from models.classification_3d import Resnet50_3d
import torch
import nibabel
from utils import *
from albumentations.pytorch.transforms import ToTensorV2
from albumentations import *
from heart_utils import *
import cv2
import statistics

def load_models(models_folder, model_configs,device):
    
    plans = {}
    state_dicts = {}
    models = {}
    for model_config in model_configs:
        model_name = model_config['name']
        model_type = model_config['type']
        print(f'Loading model:{model_name}')
        folds = model_config['folds']
        models[model_name] = []
        print(model_name)

        for i, fold in enumerate(folds):

            if model_type == 'segmentation':

                model = Segnet(num_features=1, num_outputs=4, channels=[16, 32, 64, 128], num_residual=2).to(device)
                model_path = f'{models_folder}/segnet_fold{i+1}.pth'
                model.load_state_dict(torch.load(f'{model_path}',map_location=device))
                models[model_name].append(model.eval())

            elif model_type == 'classification':

                model = Resnet50_3d().to(device)
                model_path = f'{models_folder}/diagnosis_fold{i+1}.pth'
                model.load_state_dict(torch.load(f'{model_path}',map_location=device))
                models[model_name].append(model.eval())

        print(f'{model_name} loaded')
    
    return {'models': models
           }

def inference(ed_path,es_path,ed_op_path,es_op_path,models,device='cpu'):
    
    for_which_classes = [(3), ]
    min_valid_object_size:dict=None
        
    # Resize = augmentations.transforms.Resize(256, 256)
    Resize = geometric.Resize(256, 256)
    val_both_aug = Compose([Resize])
    
    ed_img = 0
    es_img = 0
        
    for idx,path in enumerate([ed_path,es_path]):
        
        im1 = nibabel.load(path)
        volume_per_voxel = np.prod(im1.header.get_zooms(), dtype=np.float64)
        
        img = im1.get_fdata().astype('float32')
        
        if idx==0:
            
            op_path = ed_op_path
            ed_img = img
            
        else:

            op_path = es_op_path
            es_img = img
            
        norm_img = img/img.max()
        new_masks = []
        images = []
        final_pred = 0
        
        for i in range(norm_img.shape[2]):

            img1 = norm_img[...,i][...,None]
            augmented = val_both_aug(image=img1, mask=img1)
            img1 = augmented['image']
            to_tensor = ToTensorV2()
            img1 = np.clip(img1, 0, 1)
            img1 = (img1-0.5)/0.5
            to_tensor = ToTensorV2()
            img1 = to_tensor(image=img1)
            img1 = img1['image']
            
            for i in range(len(models)):
                
                pred = models[i](img1[None,...].to(device)).detach()
                final_pred += pred[0,...]
                
            predicted_mask = np.argmax(final_pred.detach().cpu(), 0).numpy().astype('float32')
            new_mask = cv2.resize(predicted_mask,(img.shape[1],img.shape[0]),interpolation = cv2.INTER_NEAREST)
            new_masks.append(new_mask)
            
        mask_new = np.array(new_masks)
        mask_new = np.transpose(mask_new,(1,2,0))

        mask_new,_,_ = load_remove_save(mask_new, volume_per_voxel ,for_which_classes, min_valid_object_size)
        nibabel.save(nibabel.Nifti1Image(mask_new,im1.affine),op_path)
    
    return ed_img,es_img


def get_heart_stats(ed,ed_spacing,es,es_spacing):
       
    ed_lv, ed_rv, ed_myo = heart_metrics(ed,
            ed_spacing)
    es_lv, es_rv, es_myo = heart_metrics(es,
                    es_spacing)
    ef_lv = ejection_fraction(ed_lv, es_lv)
    ef_rv = ejection_fraction(ed_rv, es_rv)
    
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
    
    heart_param = {'EDV_LV': ed_lv, 'EDV_RV': ed_rv, 'ESV_LV': es_lv, 'ESV_RV': es_rv,
        'ED_MYO': ed_myo, 'ES_MYO': es_myo, 'EF_LV': ef_lv, 'EF_RV': ef_rv,
        'ES_MYO_MAX_AVG_T': es_myo_thickness_max_avg, 'ES_MYO_STD_AVG_T': es_myo_thickness_std_avg, 'ES_MYO_AVG_STD_T': es_myo_thickness_mean_std, 'ES_MYO_STD_STD_T': es_myo_thickness_std_std,
        'ED_MYO_MAX_AVG_T': ed_myo_thickness_max_avg, 'ED_MYO_STD_AVG_T': ed_myo_thickness_std_avg, 'ED_MYO_AVG_STD_T': ed_myo_thickness_mean_std, 'ED_MYO_STD_STD_T': ed_myo_thickness_std_std,}
    
    heart_stats =[]

    heart_stats.append(heart_param['EDV_LV'])
    heart_stats.append(heart_param['ESV_LV'])
    heart_stats.append(heart_param['EDV_RV'])
    heart_stats.append(heart_param['ESV_RV'])
    heart_stats.append(heart_param['ED_MYO'])
    heart_stats.append(heart_param['ES_MYO'])
    heart_stats.append(heart_param['EF_LV'])
    heart_stats.append(heart_param['EF_RV'])
    heart_stats.append(ed_lv/ed_rv)
    heart_stats.append(es_lv/es_rv)
    heart_stats.append(ed_myo/ed_lv)
    heart_stats.append(es_myo/es_lv)
    # r.append(patient_data[pid]['Height'])
    # r.append(patient_data[pid]['Weight'])
    heart_stats.append(heart_param['ES_MYO_MAX_AVG_T'])
    heart_stats.append(heart_param['ES_MYO_STD_AVG_T'])
    heart_stats.append(heart_param['ES_MYO_AVG_STD_T'])
    heart_stats.append(heart_param['ES_MYO_STD_STD_T'])

    heart_stats.append(heart_param['ED_MYO_MAX_AVG_T'])
    heart_stats.append(heart_param['ED_MYO_STD_AVG_T'])
    heart_stats.append(heart_param['ED_MYO_AVG_STD_T'])
    heart_stats.append(heart_param['ED_MYO_STD_STD_T'])
    
    return heart_param, heart_stats

def load_feats(feats_path,heart_features,feature_configs,device='cpu'):
    
    train_feats = []

    for fold in feature_configs[0]['folds']:

        train_feat_means = np.load(f'{feats_path}/train_feat_means_fold_{fold}.npy')
        train_feat_stds = np.load(f'{feats_path}/train_feat_stds_fold_{fold}.npy')
        feats = (heart_features - train_feat_means) / train_feat_stds
        train_feats.append((torch.from_numpy(feats).to(device).float()))

    return train_feats

def inference_classification(ed_img,es_img,ed,es,models,features,device='cpu'):
    

    img = np.concatenate([
    ed_img[None, ...], 
    es_img[None, ...],
    ed[None, ...],
    es[None, ...]
    ]).astype('float32')
    
    img[:2, ...] = img[:2, ...] / 1000 # scale image
    crop = CenterCrop_new(size=(312, 312, 16))
    img = crop(img)
    img = torch.from_numpy(img).to(device)
    
    img = img[None, ...]
    
    predictions = []
    
    for idx,feature in enumerate(features):
        
        
        feature = feature[None, ...]
        with torch.no_grad():
            prediction = models[0](img, feature)
        predictions.append(prediction.argmax(1).item())
    
    
    return(statistics.mode(predictions))

