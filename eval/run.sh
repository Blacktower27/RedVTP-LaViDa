set -x

mkdir -p logs 
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
LOGFILE="logs/run_${timestamp}.log"
exec > >(tee -a "$LOGFILE") 2>&1
echo "🔹 Logging to $LOGFILE"
echo "Start time: $(date)"

LLADA_VISION_ENCODER="google/siglip-so400m-patch14-384"

set -x
# TASKS=
# export TASKS=${TASKS:-"mme,vqav2_val_lite,mmbench_en_dev_lite,textvqa_val,docvqa_val,chartqa_lite,infovqa_val_lite,scienceqa_full,ai2d,coco2017_cap_val_lite,mathverse_testmini_vision_dominant,mathvista_testmini_format"}
# export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9
# export TASKS=${TASKS:-"ai2d,realworldqa,infovqa_val"}
# export TASKS=${TASKS:-"docvqa_val,mme,mmbench_en_dev"}
export TASKS=${TASKS:-"ai2d"}
export CUDA_VISIBLE_DEVICES=0,
export DEBUG_PRINT_IMAGE_RES=1
echo $TASKS

accelerate launch --num_processes=1 \
    -m lmms_eval \
    --model llava_llada \
    --model_args pretrained=$1,conv_template=llada,model_name=llava_llada \
    --tasks $TASKS \
    --batch_size 1 \
    --gen_kwargs prefix_lm=False \
    --log_samples \
    --log_samples_suffix llava_llada \
    --output_path ./logs/ --verbosity=DEBUG \
    ${@:2} \
