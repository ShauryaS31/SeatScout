# SeatScout – Real-Time Seat Occupancy Monitor (UTS Library)

SeatScout is an innovative project that provides a real-time seat occupancy monitoring system, specifically developed for the UTS Library. This system leverages advanced computer vision techniques to accurately detect seat occupancy using CCTV feeds.

## Features
- YOLOv12-Based Detection: Utilizes a custom YOLOv12 model for highly accurate seat occupancy detection.
- High Precision: Achieved 97% average precision on test data, demonstrating robust performance.
- Real-Time Dashboard: Deploys a user-friendly, real-time dashboard built with Streamlit for live monitoring.

Custom Dataset: Developed and annotated a unique synthetic dataset using ChatGPT and Hugging Face Stable Diffusion for data generation, and CVAT for annotation.

UTS AI Showcase Participant: Selected to present at the UTS AI Showcase, recognizing it as a top project among 100+ submissions.

Pitch Video
Watch our Pitch video here - https://www.youtube.com/watch?v=gv9jatp9rOk

Setup and Installation
To run SeatScout, you need to have Python installed on your system, along with Streamlit.

Prerequisites:

Python (3.x recommended)
Streamlit

Install Dependencies:
It is recommended to create a virtual environment first.

How to Run
Once you have the prerequisites and dependencies installed, you can run the dashboard:
```bash
streamlit run dashboard.py
```
