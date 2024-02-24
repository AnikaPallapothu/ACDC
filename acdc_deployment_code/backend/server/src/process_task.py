import yaml
import os
import nibabel
from inference_utils import *
import uuid
import sys
import datetime
import pymongo
import time
import redis
import create_obj

def process_images(
    ed_path,
    es_path,
    uuid=None,
    nii_file=True
    ):


    config = yaml.load(open('config.yaml'), yaml.Loader)
    models_folder = f"{config['models_folder']}"
    features_folder = f"{config['features_folder']}"
    models_configs = config['models_configs']
    feature_configs = config['feature_configs']
    device = config['device']
    labels = config['labels']
    mongo_creds = config['mongo_creds'][0]
    payloads_folder = config['payloads_folder']

    client = pymongo.MongoClient(mongo_creds['service'], mongo_creds['port'], username=mongo_creds['username'], password=mongo_creds['password'])
    db = client[mongo_creds['db']]
    col = db[mongo_creds['col']]

    if uuid is None:
        uuid = uuid.uuid4().hex

    print('uuid: ', uuid, flush=True)
    print('Loading models................', flush=True)

    models = load_models(models_folder, models_configs,device)

    print('Finished Loading models................', flush=True)

    patient_identity = get_dummy_patient_id(unique_id=uuid)

    # col.insert_one(patient_identity)

    # patient_name = ed_path.split('/')[-1].split('.nii.gz')[0].split('_')[0]
    record = col.find_one({"uuid": uuid})
    if record is not None:
        patient_name = record.get('patient_name', 'Not reported')
        patient_id = record.get('patient_id', 'Not reported')
        patient_age = record.get('patient_age', 'Not reported')
        patient_sex = record.get('patient_sex', 'Not reported')
        patient_weight = record.get('patient_weight', 'Not reported')
        patient_height = record.get('patient_height', 'Not reported')
        
        patient_identity['patient_name'] = patient_name
        patient_identity['patient_id'] = patient_id
        patient_identity['patient_age'] = patient_age
        patient_identity['patient_sex'] = patient_sex
        patient_identity['patient_weight'] = patient_weight
        patient_identity['patient_height'] = patient_height



    if ed_path.split('/')[-1].endswith('.nii.gz'):
        nii_file = True

    if not nii_file:
        dicoms_paths =  key if not isinstance(key, str) else get_dicom_paths(key)
        for dicom_path in dicoms_paths:
            try:
                img_info = pydicom.dcmread(dicom_path)
                break
            except Exception as e:
                print(e)
                continue
        patient_identity = update_patient_info(patient_identity, img_info, uuid=uuid)

    elif record is None:

        patient_identity['patient_name'] = ed_path.split('/')[-1].split('.nii.gz')[0].split('_')[0]
        patient_identity['patient_id'] = ed_path.split('/')[-1].split('.nii.gz')[0].split('_')[0]

    col.update_one({"uuid": uuid},{'$set': {'patient_identity':patient_identity}})
    col.update_one({"uuid": uuid}, {"$set":{"predictionStarted": datetime.datetime.now(), "current_status": "Preprocessing"}})

    patient_name = patient_identity['patient_name']
    patient_id = patient_identity['patient_id']

    base_folder = f"{payloads_folder}/{patient_name}_{patient_id}_{uuid}"
    print('base_folder', base_folder, patient_name, patient_id)

    maybe_mkdir(base_folder)

    nifti_folder_input = f"{base_folder}/input"
    maybe_mkdir(nifti_folder_input)

    nifti_folder_output = f"{base_folder}/output"
    maybe_mkdir(nifti_folder_output)

    logs_file = f"{base_folder}/{patient_name}_{patient_id}_{uuid}.txt"

    def print_log(text, log_type='info', time_stamp=False):
        print_log_txt(logs_file, text, log_type=log_type, time_stamp=time_stamp)

    col.update_one({")ud": uuid}, {"$set": {
        "base_folder": base_folder,
        "nifti_folder_input": nifti_folder_input,
        "nifti_folder_output": nifti_folder_output,
        'logs_file': logs_file
    }})

    print(f"Received Patient: {patient_name}'s Data")
    print_log_txt(logs_file,f'Patient Details in {patient_identity}','info',True)

    #Process the input data
    dimensions = None
    orientations = None

    input_nii_path_ed = f"{nifti_folder_input}/input_ed.nii.gz"
    input_nii_path_es = f"{nifti_folder_input}/input_es.nii.gz"
    nii_image = None

    if not nii_file:
        print(f"Cant process dicom files")
        print_log_txt(logs_file,f'Cant process dicom files','info',True)

    else:

        os.system(f"cp '{ed_path}' '{input_nii_path_ed}'")
        os.system(f"cp '{es_path}' '{input_nii_path_es}'")

    ed_op_path = f'{nifti_folder_output}/ed.nii.gz'
    es_op_path = f'{nifti_folder_output}/es.nii.gz'

    col.update_one({"uuid": uuid}, {"$set":{"current_status": "Inferencing"}})

    ed_img,es_img = inference(ed_path,es_path,ed_op_path,es_op_path,models['models']['segnet'],device)

    print(f'Finished inferencing segmentation model')
    print_log_txt(logs_file,f'Finished inferencing segmentation model','info',True)

    col.update_one({"uuid": uuid}, {"$set":{"current_status": "Postprocessing"}})

    # create stl using vtk
    print_log_txt(logs_file,f'Creating stl files','info',True)
    create_obj.create_obj(ed_op_path, config['segmentation_labels'])
    create_obj.create_obj(es_op_path, config['segmentation_labels'])
    

    ed_img_mask = nibabel.load(ed_op_path)
    ed_spacing = ed_img_mask.header.get_zooms()
    ed = ed_img_mask.get_fdata()

    es_img_mask = nibabel.load(es_op_path)
    es_spacing = es_img_mask.header.get_zooms()
    es = es_img_mask.get_fdata()

    print(f'Getting stats of cardiac features')
    print_log_txt(logs_file,f'Getting stats of cardiac features','info',True)

    heart_param, heart_feats = get_heart_stats(ed,ed_spacing,es,es_spacing)

    print(f'Finished getting stats of cardiac features')
    print_log_txt(logs_file,f'Finished getting stats of cardiac features','info',True)

    print(f'heart_feats: {heart_feats}')
    print_log_txt(logs_file,f'heart_feats: {heart_feats}','info',True)


    heart_feats =  np.array(heart_feats[:21]).astype('float32')
    features = load_feats(features_folder,heart_feats,feature_configs,device)

    final_prediction = inference_classification(ed_img,es_img,ed,es,models['models']['resnet3d'],features,device)

    print(f'Finished inferencing classification model')
    print_log_txt(logs_file,f'Finished inferencing classification model','info',True)

    patient = patient_identity.copy()
    patient['diagnosis'] = f"{patient_name} found to have {labels[final_prediction]}"
    # make report 
    pdf_path = f'{nifti_folder_output}/report.pdf'
    make_report(patient, heart_param, pdf_path)

    col.update_one({
        "uuid": uuid
    }, {
        "$set": {
            "current_status": "Finished",
            "model_outputs": {
                "ed": ed_op_path,
                "es": es_op_path,
            },
            "findings": labels[final_prediction],
            "report_path": pdf_path,
            "completed_at": datetime.datetime.now(),
        }
    })

    print(f'Patient belongs to:{labels[final_prediction]} class')
    print_log_txt(logs_file,f'Patient belongs to:{labels[final_prediction]} class','info',True)





if __name__ == '__main__':


    if len(sys.argv) == 3:

        ed_path = sys.argv[-2]
        es_path = sys.argv[-1]

        print('\nProcessing images.....\n')
        process_images(ed_path,es_path)
        print('\nFinished running script\n')

        # while(1):

        #     print('yes')
        #     time.sleep(200)