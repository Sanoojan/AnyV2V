import frechet_video_distance as fvd
import os
from natsort import natsorted

real_videos_path="Edited_frames/SimSwap"
generated_videos_path="Edited_frames/SimSwap"
real_video_batch_size=16
generated_videos_batch_size=16
number_of_frames=16

video_names = os.listdir(real_videos_path)

real_video_list=[]
for i in range(real_video_batch_size):
    video_names[i] = os.path.join(real_videos_path, video_names[i])
    real_video_list.append(video_names[i])
    # read #number_of_frames frames from each video and make as tensorflow tensors
    


result = fvd.calculate_fvd(
    fvd.create_id3_embedding(fvd.preprocess(real_videos, (224, 224))),
    fvd.create_id3_embedding(fvd.preprocess(generated_videos, (224, 224))))