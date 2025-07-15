
# Real-time Football Player Tracking with YOLOv11 and ByteTrack

## Project Overview

This project implements a robust real-time multi-object tracking system designed to identify and consistently track individual football players within a video feed. The core objective is to ensure accurate re-identification of players, meaning that each player maintains a unique ID even if they leave the frame and re-enter, simulating a real-world tracking scenario.

## Features

* **Player Detection:** Utilizes a fine-tuned YOLOv11 model for accurate player detection in each video frame.
* **Persistent ID Tracking:** Employs a state-of-the-art tracking algorithm (ByteTrack via Ultralytics' integrated tracking) to assign and maintain consistent IDs for players across frames, crucial for re-identification.
* **Video Processing:** Reads input video and generates an annotated output video with bounding boxes and unique player IDs.
* **Real-time Simulation:** Designed for efficient processing to simulate real-time performance.

## Technologies Used

* **Python:** The primary programming language.
* **Ultralytics YOLOv11:** For highly accurate object detection.
* **OpenCV (`cv2`):** For video input/output operations and drawing visualizations.
* **PyTorch (`torch`):** Underlying deep learning framework for YOLOv11.
* **ByteTrack (via `ultralytics.track`):** The robust tracking algorithm used for association and re-identification (leveraged by Ultralytics when `supervision` library is installed).

## Setup and Installation

Follow these steps to set up the project environment and run the code.

### Prerequisites

* Python.
* Git (for cloning repositories).

### 1. Clone the Repository

Start by cloning this repository to your local machine:


git clone [https://github.com/Sri-Naimisha/Player-Tracking-ReID](https://github.com/Sri-Naimisha/Player-Tracking-ReID)

cd Player-Tracking-ReID


### 2\. Create and Activate a Virtual Environment (Recommended)

It's highly recommended to use a virtual environment to manage project dependencies and avoid conflicts with other Python projects.

```bash
python -m venv venv
```

  * **On Windows:**
    ```bash
    .\venv\Scripts\activate
    ```
  * **On macOS/Linux:**
    ```bash
    source venv/bin/activate
    ```
    You should see `(venv)` prepended to your terminal prompt, indicating the virtual environment is active.

### 3\. Install Dependencies

Install all required Python packages using `pip`.

```bash
pip install ultralytics opencv-python torch torchvision supervision
```

*(Note: `torch` might require specific installation instructions based on your CUDA version if you plan to use GPU. Refer to [PyTorch's official website](https://pytorch.org/get-started/locally/) for details.)*

### 4\. Download Project Assets

The project requires the pre-trained YOLOv11 model and an input video. These large files are not included directly in the repository but can be downloaded from Google Drive.

  * **YOLOv11 Model (`best.pt`):**
    Download the model file from this link:
    
    [https://drive.google.com/file/d/1-5fOSHOSB9UXyP\_enOoZNAMScrePVcMD/view](https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view)
    
    **Place the downloaded `best.pt` file in the root directory of this project.**

  * **Input Video (`15sec_input_720p.mp4`):**
    Download the input video from this link:
    
    [https://drive.google.com/file/d/1TDcND31fvEDvcnZCaianTxJrmT8q7iIi/view?usp=drive\_link](https://drive.google.com/file/d/1TDcND31fvEDvcnZCaianTxJrmT8q7iIi/view?usp=drive_link)
    
    **Place the downloaded `15sec_input_720p.mp4` file in the root directory of this project.**

## How to Run the Code

Once all dependencies are installed and the model and input video files are placed in the root directory:

1.  **Ensure your virtual environment is active.**

2.  **Run the `tracking.py` script:**

    ```bash
    python tracking.py
    ```

3.  The script will process the input video and save the tracked output to a new file named `output_tracked.avi` in the same directory.

## Project Output

The generated `output_tracked.avi` video showcases the real-time player tracking with persistent unique IDs.

  * **View the Output Video:**
    [https://drive.google.com/file/d/1z4oCO0WHBd8P1P37ymrzq7vpbN2dnIB4/view?usp=sharing](https://drive.google.com/file/d/1z4oCO0WHBd8P1P37ymrzq7vpbN2dnIB4/view?usp=sharing)

