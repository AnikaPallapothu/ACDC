from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends
from fastapi import responses
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.wsgi import WSGIMiddleware
from typing import List
from pydantic import BaseModel
# from bson import ObjectId
import process_task
import datetime 
import yaml
import redis
import pymongo
import uuid
import os 
import shutil
import cv2
import nibabel as nib
import SimpleITK as sitk
import numpy as np
import io 

import utils

from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.security.oauth2 import OAuth2PasswordBearer
import bcrypt
import jwt
import time

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:56495",
    "http://localhost:61341"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)

redis_client = redis.Redis(host='redis_service', port=6379, db=0)
mongo_creds = config['mongo_creds'][0]
mongo_client = pymongo.MongoClient(mongo_creds['service'], mongo_creds['port'], username=mongo_creds['username'], password=mongo_creds['password'])
mongo_db = mongo_client[mongo_creds['db']]
mongo_col = mongo_db[mongo_creds['col']]

@app.get('/health')
async def health():
    return 'OK'

def create_token(username: str):
    payload = {
        'username': username,
        'expires': int(time.time()) + 3600,
    }
    token = jwt.encode(payload, config['jwt_secret'], algorithm='HS256')
    return token

@app.post('/login')
async def login(username: str = Form(...), password: str = Form(...)):
    try:

        user = mongo_db.users.find_one({'username': username})
        if user is None:
            return {'status': 'error', 'message': 'User not found'}

        # check if user is verified
        if 'verified' not in user or user['verified'] == False:
            return {'status': 'error', 'message': 'User not verified'}

        if bcrypt.checkpw(password.encode('utf-8'), user['password']):
            token = create_token(username)
            return {'status': 'success', 'message': 'Login successful', 'token': token}

        return {'status': 'error', 'message': 'Invalid password'}

    except Exception as e:
        return {'status': 'error', 'message': str(e)}


@app.post('/register')
async def register(username: str = Form(...), password: str = Form(...)):
    try:
        user = mongo_db.users.find_one({'username': username})
        if user is not None:
            return {'status': 'error', 'message': 'User already exists'}

        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        mongo_db.users.insert_one({'username': username, 'password': hashed_password, 'verified': False})

        verification_code = ''.join([ uuid.uuid4().hex for _ in range(8) ])

        redis_client.set('verification_code_' + username, verification_code)

        utils.send_verification_mail(username, verification_code)

        return {'status': 'success', 'message': 'Registration successful, please verify your email address'}

    except Exception as e:
        return {'status': 'error', 'message': str(e)}

@app.get('/verify')
async def verify(username: str, code: str):
    try:
        if redis_client.get('verification_code_' + username).decode('utf-8') == code:
            # delete verification code
            redis_client.delete('verification_code_' + username)
            # update user as verified
            mongo_db.users.update_one({'username': username}, {'$set': {'verified': True}})
            return {
                'status': True,
                'message': 'User verified'
            }, 200
        else:
            return {
                'status': False,
                'error': 'Invalid verification code'
            }, 200
    except Exception as e:
        print(e)
        return {
            'status': False,
            'error': str(e)
        }, 500

def authenticate(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, config['jwt_secret'], algorithms=['HS256'])
        if payload['expires'] < int(time.time()):
            return {'status': 'error', 'message': 'Token expired'}
        return {'status': 'success', 'payload': payload}
    except jwt.ExpiredSignatureError:
        return {'status': 'error', 'message': 'Token expired'}
    except jwt.InvalidTokenError:
        return {'status': 'error', 'message': 'Invalid token'}

@app.post('/logout')
async def logout():
    return 'OK'

# upload file to server
@app.post('/upload')
async def upload(
    ed: UploadFile = File(...), 
    es: UploadFile = File(...), 
    patient_id: str = Form(...),
    patient_name: str = Form(...),
    patient_age: str = Form(...),
    patient_sex: str = Form(...),
    patient_height: str = Form(...),
    patient_weight: str = Form(...),    
    current_user: str = Depends(authenticate)):
    try:
        print('current_user: ', current_user, flush=True)
        print('patient_id: ', patient_id, flush=True)
        if 'payload' not in current_user:
            return {'status': 'error', 'message': 'Unauthorized access'}
        ed_file = await ed.read()
        es_file = await es.read()
        UUID = uuid.uuid4().hex
        os.makedirs(os.path.join(config['uploads_folder'], UUID))
        with open(os.path.join(config['uploads_folder'], UUID, 'ed.nii.gz'), 'wb') as f:
            f.write(ed_file)
        with open(os.path.join(config['uploads_folder'], UUID, 'es.nii.gz'), 'wb') as f:
            f.write(es_file)

        mongo_col.insert_one({
            'uuid': UUID,
            'ed_file_path': os.path.join(config['uploads_folder'], UUID, 'ed.nii.gz'),
            'es_file_path': os.path.join(config['uploads_folder'], UUID, 'es.nii.gz'),
            'patient_name': patient_name,
            'patient_id': patient_id,
            'patient_age': patient_age,
            'patient_sex': patient_sex,
            'patient_height': patient_height,
            'patient_weight': patient_weight,
            'received_at': datetime.datetime.now(),
            'username': current_user['payload']['username'],
        })
        return {'status': 'success', 'message': 'File uploaded successfully', 'uuid': UUID}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

# trigger prediction
@app.post('/predict')
async def predict(UUID: str = Form(...), background_tasks: BackgroundTasks = BackgroundTasks(), current_user: str = Depends(authenticate)):
    try:
        if 'payload' not in current_user:
            return {'status': 'error', 'message': 'Unauthorized access'}
        record = mongo_col.find_one({'uuid': UUID})
        if record is None:
            return {'status': 'error', 'message': 'Record not found'}
        # check if record is of current user
        if record['username'] != current_user['payload']['username']:
            return {'status': 'error', 'message': 'Unauthorized access'}
        background_tasks.add_task(process_task.process_images, record['ed_file_path'], record['es_file_path'], UUID)
        redis_client.set('task_status_' + UUID, 'started')
        mongo_col.update_one({'uuid': UUID}, {'$set': {'current_status': 'Queued'}})
        return {'status': 'success', 'message': 'Prediction triggered successfully'}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

# get all files from server
@app.get('/records')
async def records(page: int = 0, limit: int = 10, current_user: str = Depends(authenticate)):
    try:
        if 'payload' not in current_user:
            return {'status': 'error', 'message': 'Unauthorized access'}
        query = { 'username': current_user['payload']['username'] }
        records_cursor = mongo_col.find(query, {
            '_id': 0, 
        }).skip(page * limit).limit(limit)
        records = [record for record in records_cursor]
        print('records: ', records, flush=True)
        return {'status': 'success', 'records': records}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

# get file from server
@app.get('/record/<record_id>')
async def record(record_id: str, current_user: str = Depends(authenticate)):
    try:
        record = mongo_col.find_one({'uuid': record_id})
        return {'status': 'success', 'record': record}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

@app.delete('/record/{uuid}')
async def delete_record(uuid: str, current_user: str = Depends(authenticate)):
    try:
        if 'payload' not in current_user:
            return {'status': 'error', 'message': 'Unauthorized access'}
        record = mongo_col.find_one({'uuid': uuid, 'username': current_user['payload']['username']})
        print('record: ', record, flush=True)
        if record is None:
            return {'status': 'error', 'message': 'Record not found'}
        if 'patient_identity' in record:
            patient_name = record['patient_identity']['patient_name']
            patient_id = record['patient_identity']['patient_id']
        mongo_col.delete_one({'uuid': uuid})
        # delete from payload folder
        if 'patient_identity' in record:
            payload_folder = f"{patient_name}_{patient_id}_{uuid}"
        shutil.rmtree(os.path.join(config['payloads_folder'], payload_folder))
        return {'status': 'success', 'message': 'Record deleted successfully'}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

@app.get('/image_metadata/{UUID}')
async def get_image(UUID: str, current_user: str = Depends(authenticate)):
    try:
        if 'payload' not in current_user:
            return {'status': 'error', 'message': 'Unauthorized access'}
        record = mongo_col.find_one({'uuid': UUID, 'username': current_user['payload']['username']})
        if record is None:
            return {'status': 'error', 'message': 'Record not found'}
        if record['current_status'] != 'Finished':
            return {'status': 'error', 'message': 'Prediction not finished yet'}
        image_path = record['es_file_path']
        img = nib.load(image_path)
        shapes = img.shape
        num_images = int(shapes[2])
        return {
            'status': 'success', 
            'message': 'Image metadata fetched successfully', 
            'metadata': {
                'num_images': num_images,
                'image_shape': shapes[:2][::-1],
                },
            'labels': config['segmentation_labels'],
            }
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

# # get image from server
@app.get('/image/{UUID}/{image_type}')
async def get_image(UUID: str, image_type: str, current_user: str = Depends(authenticate)):
    try:
        if 'payload' not in current_user:
            return {'status': 'error', 'message': 'Unauthorized access'}
        record = mongo_col.find_one({'uuid': UUID, 'username': current_user['payload']['username']})
        if record is None:
            return {'status': 'error', 'message': 'Record not found'}
        if record['current_status'] != 'Finished':
            return {'status': 'error', 'message': 'Prediction not finished yet'}
        if image_type == 'ed':
            image_path = record['ed_file_path']
            output_path = record['model_outputs']['ed']
        elif image_type == 'es':
            image_path = record['es_file_path']
            output_path = record['model_outputs']['es']
        else:
            return {'status': 'error', 'message': 'Invalid image type'}
        # img = nib.load(image_path)
        # arr = img.get_fdata()
        # arr = arr.transpose(2, 0, 1)
        img = sitk.ReadImage(image_path)
        arr = sitk.GetArrayFromImage(img)
        arr = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2])
        # label = nib.load(output_path).get_fdata().astype('uint8')
        # label = label.transpose(2, 0, 1)
        label = sitk.ReadImage(output_path)
        label = sitk.GetArrayFromImage(label)
        label = label.reshape(label.shape[0] * label.shape[1], label.shape[2])
        arr_lb = (arr % 256).astype('uint8')
        arr_ub = (arr / 256).astype('uint8')
        arr = np.stack([arr_lb, arr_ub, label], axis=-1)
        print('arr: ', arr.shape, flush=True)
        # arr = np.zeros_like(arr)

        # arr[50:100, 50:100, 0] = 255
        arr_png = cv2.imencode('.png', arr)[1]
        

        return responses.StreamingResponse(io.BytesIO(arr_png.tobytes()), media_type="image/png")
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

@app.get('/obj/{UUID}/{image_type}')
async def get_image(UUID: str, image_type: str, current_user: str = Depends(authenticate)):
    try:
        record = mongo_col.find_one({'uuid': UUID, 'username': current_user['payload']['username']})
        if record is None:
            return {'status': 'error', 'message': 'Record not found'}
        if record['current_status'] != 'Finished':
            return {'status': 'error', 'message': 'Prediction not finished yet'}
        if image_type == 'ed':
            output_path = record['model_outputs']['ed']
        elif image_type == 'es':
            output_path = record['model_outputs']['es']
        else:
            return {'status': 'error', 'message': 'Invalid image type'}
        obj_path = output_path.replace('.nii.gz', '.obj')
        with open(obj_path, 'r') as f:
            obj = f.read()
        mtl_path = output_path.replace('.nii.gz', '.mtl')
        with open(mtl_path, 'r') as f:
            mtl = f.read()
        data = {
            'obj': obj,
            'mtl': mtl,
        }
        return responses.JSONResponse(data)
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


# get report
@app.get('/report/{UUID}')
async def get_report(UUID: str, current_user: str = Depends(authenticate)):
    try:
        record = mongo_col.find_one({'uuid': UUID, 'username': current_user['payload']['username']})
        if record is None:
            return {'status': 'error', 'message': 'Record not found'}
        if record['current_status'] != 'Finished':
            return {'status': 'error', 'message': 'Prediction not finished yet'}
        report_path = record['report_path']
        # return pdf 
        return responses.FileResponse(report_path, media_type='application/pdf')
    except Exception as e:
        return {'status': 'error', 'message': str(e)}



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
