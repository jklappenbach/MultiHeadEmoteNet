import os
import torch
from efficientnet_pytorch import EfficientNet

def _project_root_from_this_file():
    # component/efficient_net.py -> project root is parent of 'component'
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

def load_efficientnet_weights_or_warn(model, device="gpu", strict=False):
    """
    Try to load EfficientNet weights from a set of common locations.

    Priority:
    1) EFFICIENTNET_CKPT environment variable (explicit path)
    2) Project-root typical locations (work regardless of current working dir)
    3) Current working directory typical locations
    4) Kaggle input mounts (if running on Kaggle)

    Set strict=False to proceed without loading weights if nothing is found.

    More info at: https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
    """

    # Choose the desired EfficientNet variant (e.g., 'efficientnet-b0', 'efficientnet-b1', etc.)
    model_name = 'efficientnet-b0'

    # Load a pre-trained EfficientNet model.
    # This automatically downloads the corresponding .pth file (state_dict) if not already present.
    model = EfficientNet.from_pretrained(model_name)

    # If you want to save the state_dict to a specific .pth file:
    # (The from_pretrained method handles downloading, so this is mainly for saving a modified model)
    torch.save(model.state_dict(), f'{model_name}_state_dict.pth')

    # 1) Allow explicit override via env var
    env_path = os.environ.get("EFFICIENTNET_CKPT")

    project_root = _project_root_from_this_file()
    cwd = os.getcwd()

    # 2) Candidate locations to try (ordered)
    candidate_paths = [
        env_path,
        # Project-root typical locations (stable regardless of where code is executed from)
        os.path.join(project_root, "evaluation", "checkpoints", "EfficientNet_model_state.pth"),
        os.path.join(project_root, "evaluation", "weights", "EfficientNet_model_state.pth"),
        os.path.join(project_root, "evaluation", "EfficientNet_model_state.pth"),
        os.path.join(project_root, "checkpoints", "EfficientNet_model_state.pth"),
        os.path.join(project_root, "weights", "EfficientNet_model_state.pth"),
        os.path.join(project_root, "EfficientNet_model_state.pth"),
        # Current working directory fallbacks (useful when running notebooks from subfolders)
        os.path.join(cwd, "checkpoints", "EfficientNet_model_state.pth"),
        os.path.join(cwd, "weights", "EfficientNet_model_state.pth"),
        os.path.join(cwd, "EfficientNet_model_state.pth"),
        # Kaggle-style paths (only if running in Kaggle)
        "/kaggle/input/effnet/pytorch/default/1/EfficientNet_model_state.pth",
        "/kaggle/input/effnet/EfficientNet_model_state.pth",
    ]
    candidate_paths = [p for p in candidate_paths if p]  # drop None

    found_path = next((p for p in candidate_paths if os.path.isfile(p)), None)

    if not found_path:
        msg = (
            "EfficientNet checkpoint not found. Searched:\n- "
            + "\n- ".join(candidate_paths)
            + "\nSet EFFICIENTNET_CKPT to the correct path or place the file in a known location."
        )
        if strict:
            raise FileNotFoundError(msg)
        else:
            print("⚠️ " + msg + "\nProceeding without loading weights.")
            return model.to(device).eval()

    print(f"🔹 Loading EfficientNet weights from: {found_path}")
    state = torch.load(found_path, map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()
