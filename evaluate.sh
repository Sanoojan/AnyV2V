echo "FID score with Dataset:" >> "$output_filename"

device=0
Dataset_path="path/to/dataset"
Results_out="path/to/results"



CUDA_VISIBLE_DEVICES=${device} python eval_tool/fid/fid_score.py --device cuda \
    "${Dataset_path}" \
    "${Results_out}"  >> "$output_filename"