# YOLO OBB Component and Defect Detection

This project trains and runs Ultralytics YOLO models for oriented bounding box
(OBB) detection. An OBB encloses an object with a four-corner polygon, which is
useful for angled or elongated components.

There are two workflows:

1. **Train** a model from images and OBB labels. Training produces a checkpoint
   and validation images that combine the ground-truth labels with model
   predictions.
2. **Predict** with an already trained OBB checkpoint. Inference draws
   detections on input images; it does not need ground-truth label files.

A checkpoint only detects the classes it was trained on. For another user's
classes, use a checkpoint trained for those classes.

## 1. Install

The project requires Python 3.15 or newer. The commands below are for Windows
PowerShell and should be run from the project root (the folder containing
`pyproject.toml`).

This project uses **uv** and includes `uv.lock` to keep dependency versions
consistent. In PowerShell, install uv (for example, with WinGet), then install
the required Python version and sync the project environment:

```powershell
winget install --id astral-sh.uv -e
uv python install 3.15
uv sync --locked
```

If you installed uv during this step, close and reopen PowerShell so the `uv`
command is available. Run these commands from the project root. `uv sync
--locked` creates `.venv` and installs the exact dependencies from `uv.lock`.
The first training run may download the default `yolo11m-obb.pt` starting
weights, so internet access is needed unless those weights are already
available.

Each user should run setup in their own project copy. Do not copy or share
your `.venv`; `uv sync` creates a separate environment for each user.

## 2. Prepare a dataset for training

The default dataset layout is:

```text
dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
configs/
  data.yaml
```

Every image in `images/train` and `images/val` must have a matching text file
with the same filename under the corresponding `labels` folder. Each non-empty
label line must contain a class ID and eight normalized polygon coordinates:

```text
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

Coordinates are fractions of image width/height and must be between `0` and
`1`. Class IDs in the label files must match the zero-based class names in the
dataset YAML. For a different dataset, update `configs/data.yaml` (or provide
another YAML with the same format). Keep the YAML paths relative so the project
works after it is copied to another computer.

The current OHE dataset configuration contains only classes with labeled
instances in both the train and validation splits. The labels were filtered and
remapped to match these 17 class IDs. The previous 73-class checkpoint is not
compatible with this updated class map; retrain before using the reduced set.

## 3. Train

From the project root, run:

```powershell
uv run python -m yolo_model.train
```

This uses `configs/data.yaml`, starts from `yolo11m-obb.pt`, and trains for
100 epochs at 640 pixels on CPU. To train with another dataset YAML or adjust
the run:

```powershell
uv run python -m yolo_model.train `
  --data configs\my_data.yaml `
  --model yolo11m-obb.pt `
  --epochs 100 `
  --imgsz 640 `
  --batch 1 `
  --device cpu `
  --name my_run `
  --conf 0.25
```

Training checks that train and validation images have labels and that the OBB
label contents are valid. The best checkpoint is saved under
`runs\obb\<name>\weights\best.pt`. After training, the model is evaluated on
the validation split. Combined ground-truth and prediction overlays are saved
under `predictions\<name>\predictions`; validation metrics are in
`predictions\<name>\metrics.txt`.

Ground truth is needed for training and validation. It is not needed to run
inference on new images.

## 4. Run inference

Use the trained OBB checkpoint for the classes you want to detect. The user
must have that `.pt` weights file; the Python code by itself is not a trained
detector. The checkpoint must include the target classes, and results can vary
with camera, lighting, distance, angle, image quality, and scene conditions
that differ from its training data. Test it on representative images before
relying on results.

Run the command from the project root (the folder containing `pyproject.toml`).
Give the first argument as the weights-file path and the second as an image,
video, or folder path. Paths can be relative to the project root or absolute.
In PowerShell, quote paths that contain spaces:

```powershell
# One image, using paths relative to the project root
uv run python src\yolo_model\predict.py `
  "C:\Users\OtherUser\models\best.pt" `
  "C:\Users\OtherUser\Pictures\inspection 01.jpg" `
  0.25

# A folder of images
uv run python src\yolo_model\predict.py `
  "C:\Users\OtherUser\models\best.pt" `
  "C:\Users\OtherUser\Pictures\inspection images" `
  0.25

# A video file
uv run python src\yolo_model\predict.py `
  "C:\Users\OtherUser\models\best.pt" `
  "C:\Users\OtherUser\Videos\01071.mp4" `
  0.25
```

The confidence argument is optional; if omitted, the script uses `0.25`.
Higher thresholds usually show fewer, more confident detections. Inference
results are written to a new run-specific subfolder under `predictions` in the
current working directory. The folder name includes the input name, model name,
and run time, so separate runs do not overwrite each other's images. To choose
an output folder explicitly, add `--output path\to\my_results`.

For a folder, the script scans only that folder (not its subfolders) and
processes `.jpg`, `.jpeg`, `.png`, `.mp4`, `.avi`, `.mov`, `.mkv`, `.m4v`,
`.mpeg`, and `.mpg` files. Images are saved with their original filenames.
Videos are processed frame by frame and saved as `<video name>_predicted.mp4`.
Detected boxes use class-colored outlines and compact, background-free labels
showing the class name and confidence. A text outline helps keep labels readable
without a filled label box. Annotation size scales with the video resolution;
this improves visibility but does not improve the model's detection accuracy.
It does not need labels or use previous prediction results. Class IDs and names
come from the supplied checkpoint, so they do not have to match this project's
dataset YAML.
Prediction outputs are local and excluded from Git.

## 5. Set defect colors

Boxes for classes listed in `DEFECT_CLASSES` in
`src\yolo_model\class_colors.py` are red. Other classes are green. For another
dataset, add its exact defect class names as stored in its checkpoint to that
set. This only changes visualization colors; it does not add classes to the
model or improve detection.

## Important notes

- A model cannot detect a new class just because its name is added to
  `DEFECT_CLASSES`; it must be trained on labeled examples of that class.
- Checkpoint quality depends on the amount and quality of training data.
  Evaluate validation metrics and inspect predictions before relying on them.
- `src\yolo_model\visualize_labels.py` is a sample visualizer for this
  repository's own dataset, not a general training or inference command.

## Sharing the project

Share the source code and setup files, such as `src`, `configs`, `README.md`,
`pyproject.toml`, `uv.lock`, and `.python-version`. Do not include your
`.venv`, `dataset`, `predictions`, `runs`, or `.pt` files. They are local,
ignored by Git, and may contain your environment, data, model weights, or
results.

Each recipient creates their own `.venv` with `uv sync --locked`, supplies
their own images, and uses or trains their own compatible `.pt` checkpoint.
Prediction output is generated locally from the recipient's input images in a
new run-specific `predictions` folder; your local predictions are not needed
or used. The `.gitignore` rules do not automatically exclude files from a ZIP,
so check the archive contents before sharing it.
