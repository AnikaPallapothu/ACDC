How to run the script??

python run.py --ed_path path_to_distole_image --es path_to_systole_image --device to_run_on_cpu_or_gpu

Eg: python run.py --ed_path sample_images/patient098_ed.nii.gz --es_path sample_images/patient098_es.nii.gz --device cuda:0

Final output will be the final classification of the cardiac anamoly detected by the DL model