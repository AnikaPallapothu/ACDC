import vtk 
import numpy as np
import vtk.util.numpy_support as vtknp
import sys
import nibabel as nib
import math

colors = [
    [224, 16, 58],
    [83, 64, 223],
    [55, 182, 21],
    [128, 16, 90],
]

def create_obj(nii_path, labels):
    # create an stl file and save as .stl without rendering

    # read nifti file
    img = nib.load(nii_path)
    arr = img.get_fdata()
    zooms = img.header.get_zooms()

    arr = np.pad(arr, ((10, 10), (10, 10), (10, 10)), 'constant', constant_values=0)

    unique_values = np.unique(arr)
    unique_values = unique_values.astype('int')
    print('unique_values: ', unique_values, flush=True)

    all_faces = []
    all_verts = []
    num_verts = 0
    
    overall_map = {
        label: i for i, label in enumerate(labels)
    }
    
    for i in unique_values:
        if i == 0:
            continue
        # create a vtk file
        arr_i = np.where(arr == i, 1, 0)
        arr_i = arr_i.astype('uint8')
        
        vti = vtk.vtkImageData()
        vti.SetDimensions(arr_i.shape[2], arr_i.shape[1], arr_i.shape[0])
        vti.SetSpacing(zooms[2], zooms[1], zooms[0])
        vti.SetOrigin(0, 0, 0)
        vti.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

        vtk_arr = vtk.util.numpy_support.numpy_to_vtk(arr_i.ravel(), deep=True)
        vtk_arr.SetName('d' + labels[i])
        vti.GetPointData().AddArray(vtk_arr)
        vti.GetPointData().SetActiveScalars('d' + labels[i])


        contour = vtk.vtkMarchingCubes()
        contour.SetInputData(vti)
        contour.SetNumberOfContours(1)
        contour.SetValue(0, 1)
        contour.Update()


        smoothingFilter = vtk.vtkWindowedSincPolyDataFilter()
        smoothingFilter.SetInputConnection(contour.GetOutputPort())
        smoothingFilter.SetNumberOfIterations(20)
        smoothingFactor = 1.0
        passBand = math.pow(10.0, -4.0 * smoothingFactor)
        smoothingFilter.SetPassBand(passBand)

        smoothingFilter.SetBoundarySmoothing(False)
        ##            smoothingFilter.NormalizeCoordinatesOn()
        smoothingFilter.Update()

        # output = contour.GetOutput()
        output = smoothingFilter.GetOutput()


        verts = vtknp.vtk_to_numpy(output.GetPoints().GetData())
        normals = vtknp.vtk_to_numpy(output.GetPointData().GetNormals())
        faces = vtknp.vtk_to_numpy(output.GetPolys().GetData())
        values = vtknp.vtk_to_numpy(output.GetPointData().GetScalars())

        faces = faces.reshape((-1, 4))[:, 1:]

        print(len(verts), len(faces), len(normals), len(values))
        print(verts.shape, faces.shape, normals.shape, values.shape)
        print(np.unique(values, return_counts=True))

        faces = faces+1
        # print('faces', faces)

        all_faces.append(faces)
        all_verts.append(verts)

    obj_file = open(nii_path.replace('.nii.gz', '.obj'), 'w')
    obj_file.write('# OBJ file\n')
    obj_file.write('# Created by vtk2obj\n')

    for verts in all_verts:
        if isinstance(verts, int) and verts == 0: continue
        for item in verts:
            obj_file.write("v {0} {1} {2}\n".format(item[0], item[1], item[2]))


    for key, val in overall_map.items():
        if val == 0: continue
        # faces1 = [ face for face in faces if values[face[0]-1] == val ]
        faces = all_faces[val - 1]
        verts = all_verts[val - 1]
        obj_file.write(f'o {key}\n')
        obj_file.write(f'usemtl {key}\n')
        if isinstance(verts, int) and verts == 0: continue
        faces += num_verts
        num_verts += len(verts)
        if isinstance(faces, int) and faces == 0: continue
        print( val, 'faces')
        for item in faces:
            obj_file.write("f {0} {1} {2}\n".format(item[0],item[1],item[2]))

    obj_file.close()

    mtl_file = open(nii_path.replace('.nii.gz', '.mtl'), 'w')
    mtl_file.write('# MTL file\n')
    mtl_file.write('# Created by vtk2obj\n')

    for key, val in overall_map.items():
        if val == 0: continue
        mtl_file.write(f'newmtl {key}\n')
        # mtl_file.write(f'Ka 0.200000 0.200000 0.200000\n')
        # mtl_file.write(f'Kd 0.800000 0.800000 0.800000\n')
        # mtl_file.write(f'Ks 1.000000 1.000000 1.000000\n')
        color = colors[val - 1]
        mtl_file.write(f'Ks 0.000000 0.000000 0.000000\n')
        mtl_file.write(f'Kd {color[0]/255:0.6} {color[1]/255:0.6} {color[2]/255:0.6}\n')
        mtl_file.write(f'Ka 0.000000 0.000000 0.000000\n')
        mtl_file.write(f'\n')




    mtl_file.close()



if __name__ == '__main__':
    create_obj(sys.argv[1])