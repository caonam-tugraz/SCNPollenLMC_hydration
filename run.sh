#!/bin/bash

#### Run from command line (no Caliban)

OPT="--batchsize=32 --epochs=200 --learning_rate=0.0001 --output=output --nlayers=1 --width=64 --transform=brightness"
#CUDA_VISIBLE_DEVICES=7 python3 imgtrans_scn_pollen.py $OPT  --arch=hhnmlpb --dimensions=1
#CUDA_VISIBLE_DEVICES=7 python3 imgtrans_scn_pollen.py $OPT  --arch=hhnmlpb --dimensions=2
#CUDA_VISIBLE_DEVICES=7 python3 imgtrans_scn_pollen.py $OPT  --arch=hhnmlpb --dimensions=3
#CUDA_VISIBLE_DEVICES=7 python3 imgtrans_scn_pollen.py $OPT  --arch=hhnmlpb --dimensions=5
#CUDA_VISIBLE_DEVICES=7 python3 imgtrans_scn_pollen.py $OPT  --arch=hhnmlpb --dimensions=8
#CUDA_VISIBLE_DEVICES=7 python3 imgtrans_one4all_pollen.py $OPT --arch=mlpb
#CUDA_VISIBLE_DEVICES=7 python3 imgtrans_inverse_pollen.py $OPT --arch=mlpb
#CUDA_VISIBLE_DEVICES=7 python3 imgtrans_one4one_pollen.py $OPT --arch=mlpb
#CUDA_VISIBLE_DEVICES=7 python3 linearconnect.py $OPT --arch=mlpb
python3 plot.py