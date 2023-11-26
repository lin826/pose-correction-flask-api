# Pose Correction

Real-time feedback on users workout form using pose correction ML models. This backend server is ready to be deployed in to WSGI server in Docker.

> [MotionCorrection](https://github.com/huguesvinzant/Motion-Correction) is able to correct movement mistakes in 3D pose sequences and output the corrected motion. This model is integrated in a pipeline containing a state-of-the-art 3D human pose estimator to go from raw video images to a sequence of corrected 3D poses. The dataset contains videos, 2D and 3D poses of correct and incorrect executions of different movements that are SQUATS, lunges, planks and pick- ups and labels identifying the mistake in each practice of that exercise.


## Prerequisites

### [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html)

```sh
conda create --name <VENV_NAME> python=3.8 -y
conda activate <VENV_NAME>
```

## Local Installation and Run

```sh
pip install -r requirements.txt
bash setup.sh
python main.py --host <HOST_IP> --port <PORT> --size <MULTITHREAD_SIZE>
```

## Docker Deployment

```sh
docker build -t pose-correction . -f Dockerfile
docker run -p 8000:8000 --name pytorch-container --gpus all pose-correction
```

## cURL Test

Either image or video can use the same script:

```sh
curl -v -F tmp_file=@/<FILE_PATH> -k http://127.0.0.1:8000/predict
```
The expected result is a JSON object with the following structure:
```
{"pose": Detected pose, "result": Is the pose correct or not}
```
For example, you can download an example video of [correct lunge](https://github.com/huguesvinzant/Motion-Correction/blob/master/PoseCorrection/Data/Videos/LUNGE_C.mp4) and run the following command
```sh
curl -v -F tmp_file=@./LUNGE_C.mp4 -k http://127.0.0.1:8000/predict
```
You should see the following result.
```
{"pose": "LUNGE", "result": "correct"}
```
