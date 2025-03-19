
echo "--K $1  --split $2"
rm -r ../show-dir_pkl/frame_detections.pkl

#echo "RUN tools/test.py"
#python tools/test.py configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_GODsingleframe.py "/media/calayzhou/TOSHIBA EXT/Project/TOP1_GOD/singleframeGODresult/cascade_rcnn_r50_fpn_1x_GODsingleframe/epoch_12.pth"   --show-dir show-dir

#echo "cd ACT_utils"

#cd ACT_utils
echo "ACT 0.5"
python ACTv2.py  --task frameAP --K $1  --th 0.5 --inference_dir ../show-dir_pkl/ --dataset IODVideo --split $2
echo "ACT 0.75"
python ACTv2.py  --task frameAP --K $1  --th 0.75 --inference_dir ../show-dir_pkl/ --dataset IODVideo --split $2
echo "ACT 0.5~0.95"
python ACTv2.py  --task frameAP_all --K $1 --inference_dir ../show-dir_pkl/ --dataset IODVideo --split $2

echo "backup TrueLeakedGas.pkl"
mv ../data/TLGDM/TrueLeakedGas.pkl  ../data/TLGDM/TrueLeakedGas_backup1.pkl

echo "AP clear"
rm -r ../show-dir_pkl/frame_detections.pkl
mv ../data/TLGDM/TrueLeakedGas_c1_290.pkl ../data/TLGDM/TrueLeakedGas.pkl
python3 ACT.py  --task frameAP_all --K $1   --inference_dir ../show-dir_pkl/ --dataset IODVideo --split $2
mv  ../data/TLGDM/TrueLeakedGas.pkl  ../data/TLGDM/TrueLeakedGas_c1_290.pkl

echo "AP vague"
rm -r ../show-dir_pkl/frame_detections.pkl
mv ../data/TLGDM/TrueLeakedGas_v1_310.pkl ../data/TLGDM/TrueLeakedGas.pkl
python3 ACT.py   --task frameAP_all --K $1   --inference_dir ../show-dir_pkl/ --dataset IODVideo --split $2
mv  ../data/TLGDM/TrueLeakedGas.pkl  ../data/TLGDM/TrueLeakedGas_v1_310.pkl

echo "restore TrueLeakedGas.pkl"
mv ../data/TLGDM/TrueLeakedGas_backup1.pkl  ../data/TLGDM/TrueLeakedGas.pkl

