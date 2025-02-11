#!/bin/bash
#source /home/YourName/miniconda3/etc/profile.d/conda.sh  #<-- change this to your own miniconda path
conda activate anyv2v-i2vgen-xl

cd ..
# python run_group_ddim_inversion.py \
# --template_config "configs/group_ddim_inversion/template_common.yaml" \
# --configs_json "configs/group_ddim_inversion/group_config_common.json" \
# --run_all \
# --samples 50

python run_group_pnp_edit.py \
--template_config "configs/group_pnp_edit/template_common.yaml" \
--configs_json "configs/group_pnp_edit/group_config_common.json" \
--run_all \
--samples 50
