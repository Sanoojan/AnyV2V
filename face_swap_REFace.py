import os
import requests
from natsort import natsorted

url = "http://127.0.0.1:5000/swap"
Target_main_folder="data/Data/VFHQ-Test/GT/Interval1_512x512_LANCZOS4"
Source_image_folder="data/Data/VFHQ-Test/Celeb_Source"
save_path_folder="Edited_frames/REFace"

number_fo_target_frames=16
naming_scheme=8
video_number=0

source_images_path=natsorted(os.listdir(Source_image_folder))
source_images_path=[os.path.join(Source_image_folder,source_images_path[i]) for i in range(len(source_images_path))]

for folder in natsorted(os.listdir(Target_main_folder)):
    print(folder)
    
    
    folder_path=os.path.join(Target_main_folder,folder)
    
    for i in range(number_fo_target_frames):
        save_path= os.path.join(save_path_folder,folder)
        if not os.path.exists(save_path):
                os.makedirs(save_path)
        save_path= os.path.join(save_path,f"%0{naming_scheme}d.png"%i)
        if os.path.exists(save_path):
            continue
        try:
            target_image_path=os.path.join(folder_path,f"%0{naming_scheme}d.png"%i)
            files = {
                "source": open(source_images_path[video_number], "rb"),
                "target": open(target_image_path, "rb"),
            }
        
            response = requests.post(url, files=files)
            
            
            
            if response.status_code == 200:
                with open(save_path, "wb") as f:
                    f.write(response.content)
                    print("Face swap successful, saved to", save_path)
            else:
                print("Error:", response.json())
        except Exception as e:
            print("Error:", e)

    video_number+=1 