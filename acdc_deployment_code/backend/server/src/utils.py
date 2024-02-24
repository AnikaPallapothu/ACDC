import ast
from copy import deepcopy
import os
import numpy as np
from scipy.ndimage import label
import SimpleITK as sitk
import shutil
import warnings
import datetime
import time
import pydicom

import jinja2
import weasyprint 
import yaml

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

warnings.filterwarnings('ignore')

def maybe_mkdir(path):
    if not os.path.exists(path):
        os.system(f"mkdir -p {path}")

def print_log_txt(log_file=None,text='',log_type='info',time_stamp=True):
    if log_file is not None:
        with open(log_file, "a") as f:
            if time_stamp:
                f.write(f"[{log_type}]:{datetime.datetime.now()}:{text}\n")
            else:
                f.write(f"                           {text}\n")

def copy_geometary(image: sitk.Image, ref: sitk.Image):
    image.SetOrigin(ref.GetOrigin())
    image.SetDirection(ref.GetDirection())
    image.SetSpacing(ref.GetSpacing())
    return image

def load_remove_save(img_npy, volume_per_voxel,for_which_classes: list,
                     minimum_valid_object_size: dict = None):
    # Only objects larger than minimum_valid_object_size will be removed. Keys in minimum_valid_object_size must
    # match entries in for_which_classes
 #   img_in = sitk.ReadImage(input_file)
#     img_npy = sitk.GetArrayFromImage(img_in)
#    volume_per_voxel = float(np.prod(spacing, dtype=np.float64))

    image, largest_removed, kept_size = remove_all_but_the_largest_connected_component(img_npy, for_which_classes, volume_per_voxel,minimum_valid_object_size)
            
    # img_out_itk = sitk.GetImageFromArray(image)
    # img_out_itk = copy_geometary(img_out_itk, img_in)
    # sitk.WriteImage(img_out_itk, op_path)
    return image, largest_removed, kept_size



def remove_all_but_the_largest_connected_component(image: np.ndarray, for_which_classes: list, volume_per_voxel: float,
                                                   minimum_valid_object_size: dict = None):
    """
    removes all but the largest connected component, individually for each class
    :param image:
    :param for_which_classes: can be None. Should be list of int. Can also be something like [(1, 2), 2, 4].
    Here (1, 2) will be treated as a joint region, not individual classes (example LiTS here we can use (1, 2)
    to use all foreground classes together)
    :param minimum_valid_object_size: Only objects larger than minimum_valid_object_size will be removed. Keys in
    minimum_valid_object_size must match entries in for_which_classes
    :return:
    """
    if for_which_classes is None:
        for_which_classes = np.unique(image)
        for_which_classes = for_which_classes[for_which_classes > 0]

    assert 0 not in for_which_classes, "cannot remove background"
    largest_removed = {}
    kept_size = {}
    for c in for_which_classes:
        if isinstance(c, (list, tuple)):
            c = tuple(c)  # otherwise it cant be used as key in the dict
            mask = np.zeros_like(image, dtype=bool)
            for cl in c:
                mask[image == cl] = True
        else:
            mask = image == c
        # get labelmap and number of objects
        lmap, num_objects = label(mask.astype(int))

        # collect object sizes
        object_sizes = {}
        for object_id in range(1, num_objects + 1):
            object_sizes[object_id] = (lmap == object_id).sum() * volume_per_voxel

        largest_removed[c] = None
        kept_size[c] = None

        if num_objects > 0:
            # we always keep the largest object. We could also consider removing the largest object if it is smaller
            # than minimum_valid_object_size in the future but we don't do that now.
            maximum_size = max(object_sizes.values())
            kept_size[c] = maximum_size

            for object_id in range(1, num_objects + 1):
                # we only remove objects that are not the largest
                if object_sizes[object_id] != maximum_size:
                    # we only remove objects that are smaller than minimum_valid_object_size
                    remove = True
                    if minimum_valid_object_size is not None:
                        remove = object_sizes[object_id] < minimum_valid_object_size[c]
                    if remove:
                        image[(lmap == object_id) & mask] = 0
                        if largest_removed[c] is None:
                            largest_removed[c] = object_sizes[object_id]
                        else:
                            largest_removed[c] = max(largest_removed[c], object_sizes[object_id])
    return image, largest_removed, kept_size

class CenterCrop_new:
    def __init__(self, size):
        self.size = size
    def __call__(self, x):
        c, oh, ow, od = x.shape
        th, tw, td = self.size
        y = x[:, max(0, (oh - th)//2): (oh + th)//2, max(0, (ow-tw)//2): (ow + tw)//2, max(0, (od-td)//2): (od+td)//2]
        y = np.pad(y, [(0, 0), (0, max(0, th-oh)), (0, max(0, tw-ow)), (0, max(0, td-od))])
        return y

def get_dicom_paths(dicom_folder):
    dicom_paths = []
    # get all dicom files
    for root, dirs, files in os.walk(dicom_folder):
        for f in files:
            # if file.endswith(".dcm"):
            if f.startswith('.'): continue
            try:
                _ = pydicom.dcmread(os.path.join(root, f), stop_before_pixels=True)
                dicom_paths.append(os.path.join(root, f))
            except Exception as e:
                print("Error: ", str(e))
    return dicom_paths

def get_dummy_patient_id(
    unique_id=None, 
    key=None
    ):
    return {
        'patient_id': 'Not reported',
        'patient_name': 'Not reported',
        'institution': 'Not reported',
        'age': 'Not reported',
        'sex': 'Not reported',
        'referring_doctor': 'Not reported',
        'unique_id': unique_id if unique_id is not None else 'Not reported',
        'study_id': 'Not reported',
        'series_id': 'Not reported',
        'study_date': 'Not reported',
        'key': key if key is not None else 'Not reported',
        'accession_number': 'Not reported',
        'patient_weight': 'Not reported',
        'patient_height': 'Not reported',
    }

def format_name_string(name):
    return str(name).replace(' ', '_').replace('^', '_').replace('%', '_').replace('/', '_')

def format_study_date(study_date):
    yy = study_date[:4]
    mm = study_date[4:6]
    dd = study_date[6:]
    return '{}/{}/{}'.format(dd, mm, yy)

def update_patient_info(patient_info, dicom, unique_id=None, key=None):
    if hasattr(dicom, 'PatientID'):
        patient_info['patient_id'] = format_name_string(dicom.PatientID)
    if hasattr(dicom, 'PatientName'):
        patient_info['patient_name'] = format_name_string(dicom.PatientName)
    if hasattr(dicom, 'PatientAge'):
        patient_info['age'] = dicom.PatientAge
    if hasattr(dicom, 'PatientSex'):
        patient_info['sex'] = dicom.PatientSex
    if hasattr(dicom, 'ReferringPhysicianName'):
        patient_info['referring_doctor'] = format_name_string(dicom.ReferringPhysicianName)
    if hasattr(dicom, 'StudyInstanceUID'):
        patient_info['study_id'] = dicom.StudyInstanceUID
    if hasattr(dicom, 'SeriesInstanceUID'):
        patient_info['series_id'] = dicom.SeriesInstanceUID
    if hasattr(dicom, 'StudyDate'):
        patient_info['study_date'] = format_study_date(dicom.StudyDate)
    if hasattr(dicom, 'AccessionNumber'):
        patient_info['accession_number'] = dicom.AccessionNumber
    if unique_id is not None:
        patient_info['unique_id'] = unique_id
    if key is not None:
        patient_info['key'] = key
    return patient_info

def make_report(patient_identity, heart_param, report_path):

    config = yaml.load(open('config.yaml'), yaml.Loader)
    environment = jinja2.Environment(loader=jinja2.FileSystemLoader(config['templates_folder']))
    template = environment.get_template(config['report_template'])

    params = {
        'LV Volume (Diastole)': round(heart_param['EDV_LV'], 2),
        'LV Volume (Systole)': round(heart_param['ESV_LV']),
        'RV Volume (Diastole)': round(heart_param['EDV_RV']),
        'RV Volume (Systole)': round(heart_param['ESV_RV']),
        'Myocardium Volume (Diastole)': round(heart_param['ED_MYO']),
        'Myocardium Volume (Systole)': round(heart_param['ES_MYO']),
        'Myocardium Thickness (Diastole)': round(heart_param['ED_MYO_MAX_AVG_T'],2),
        'Myocardium Thickness (Systole)': round(heart_param['ES_MYO_MAX_AVG_T'],2),
        'Ejection Fraction (LV)': round(heart_param['EF_LV'],2),
        'Ejection Fraction (RV)': round(heart_param['EF_RV'],2),
    }

    logo_path = 'file://' + os.path.join(config['templates_folder'], config['logo_file'])
    html = template.render(patient=patient_identity, params=params, logo_path=logo_path)

    with open(report_path.replace('.pdf', '.html'), "w") as f:
        f.write(html)

    weasyprint.HTML(string=html).write_pdf(report_path)

def send_verification_mail(email, verification_code):
    try:

        config = yaml.load(open('/src/config.yaml', 'r'), Loader=yaml.FullLoader)
        smtp = config['smtp']
  
        msg = MIMEMultipart()
        msg['Subject'] = 'ACDC SIGN UP'
        msg['From'] = smtp['username']
        msg['To'] = email

        html_content = """
        <html>
        <head></head>
        <body>
        <p>Hello {},</p>
        <p>Thank you for signing up for the <span style="font-weight: bold; color: #1a73e8;">ACDC</span> project. Please click on the link below to verify your email address.</p>
        <p style="margin-left: 20px;"><a href="{}/verify?code={}&username={}">{}/verify?code={}&username={}</a></p>
        </body>
        </html>
        """.format(email, config['hostname'], verification_code, email, config['hostname'], verification_code, email)

        msg.attach(MIMEText(html_content, 'html'))

        smtp_server = smtp['server']
        smtp_port = smtp['port']

        server = smtplib.SMTP(smtp_server, smtp_port)
        server.ehlo()
        server.starttls()
        server.login(smtp['username'], smtp['password'])

        server.send_message(msg)
        server.close()

    except Exception as e:
        print("Error: ", str(e))
