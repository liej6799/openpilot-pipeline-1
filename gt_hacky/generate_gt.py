import glob
import onnxruntime as ort
from lanes_leads_gt import generate_ground_truth
import os

data_basedir = '/home/nikita/data/'
# data_basedir = '/gpfs/space/projects/Bolt/comma_recordings/realdata/'

options = ort.SessionOptions() 
options.intra_op_num_threads = 10 
#options.inter_op_num_threads = 1

model = ort.InferenceSession('supercombo.onnx', providers=["CUDAExecutionProvider"], sess_options=options)

video_paths = glob.glob(os.path.join(data_basedir, "**/video.hevc"), recursive = True)
fcamera_paths = glob.glob(os.path.join(data_basedir, "**/fcamera.hevc"), recursive = True)

for path_to_video in video_paths:   
    print( "In processing ...", path_to_video )
   
    generate_ground_truth( path_to_video, model )
