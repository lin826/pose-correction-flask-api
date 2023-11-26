import pickle
import torch
import torch_dct as dct

from MotionCorrection.PoseCorrection.model import GCN_class
from MotionCorrection.VIBE.demo_hv import main_VIBE
from MotionCorrection.VIBE.lib.models.vibe import VIBE_Demo
from MotionCorrection.VIBE.lib.utils.demo_utils import download_ckpt
from MotionCorrection.utils.skeleton_uniform import centralize_normalize_rotate_poses


map_label = {
    0: ('SQUAT', 'Correct'), 1: ('SQUAT', 'Feets too wide'), 2: ('SQUAT', 'Knees inward'),
    3: ('SQUAT', 'Not low enough'), 4: ('SQUAT', 'Front bended'), 5: ('SQUAT', 'Unknown'),
    6: ('Lunges', 'Correct'), 7: ('Lunges', 'Not low enough'), 8: ('Lunges', 'Knees pass toes'),
    9: ('Plank', 'Correct'), 10: ('Plank', 'Banana back'), 11: ('Plank', 'Rolled back'),
}

class Vibe:
    def __init__(self, device: str):
        device = torch.device(device)
        self.model = VIBE_Demo(
            seqlen=16,
            n_layers=2,
            hidden_size=1024,
            add_linear=True,
            use_residual=True,
        ).to(device)

        pretrained_file = download_ckpt(use_3dpw=False)
        ckpt = torch.load(pretrained_file, map_location=device)
        ckpt = ckpt['gen_state_dict']
        self.model.load_state_dict(ckpt, strict=False)

        with open('MotionCorrection/PoseCorrection/Data/pose_dict.pickle', 'rb') as f:
            self.pose_dict = pickle.load(f)

    def analyze(self, video_file, img_folder=None) -> str:
        is_cuda = torch.cuda.is_available()

        # ============== 3D pose estimation ============== #
        poses = main_VIBE(video_file, self.model, img_folder=img_folder)

        # ============== Squeleton uniformization ============== #
        poses_uniform = centralize_normalize_rotate_poses(poses, self.pose_dict)
        joints = list(range(15)) + [19, 21, 22, 24]
        poses_reshaped = poses_uniform[:, :, joints]
        poses_reshaped = poses_reshaped.reshape(-1, poses_reshaped.shape[1] * poses_reshaped.shape[2]).T

        frames = poses_reshaped.shape[1]

        # ============== Input ============== #
        dct_n = 25
        if frames >= dct_n:
            inputs = dct.dct_2d(poses_reshaped)[:, :dct_n]
        else:
            inputs = dct.dct_2d(torch.nn.ZeroPad2d((0, dct_n - frames, 0, 0))(poses_reshaped))

        if is_cuda:
            inputs = inputs.cuda()

        # ============== Action recognition ============== #
        model_class = GCN_class()
        model_class.load_state_dict(torch.load('MotionCorrection/PoseCorrection/Data/model_class.pt'))

        if is_cuda:
            model_class.cuda()

        model_class.eval()
        with torch.no_grad():
            _, label = torch.max(model_class(inputs).data, 1)

        return map_label[int(label)]
