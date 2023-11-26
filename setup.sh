export PYTHONPATH=$(pwd)/MotionCorrection:$PYTHONPATH

# Setup Motion Correction Submodule
cd ./MotionCorrection/VIBE
bash ./scripts/prepare_data.sh
cd ../..
