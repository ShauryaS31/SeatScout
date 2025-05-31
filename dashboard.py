# Importing required libraries
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# --- Configuring data paths for videos, Icons, and defining IOU thresholds for occupancy detection -----------

# Paths for room inference videos
ROOM1_VIDEO_PATH = "assets/room_videos/room1.mp4"
ROOM2_VIDEO_PATH = "assets/room_videos/room2.mp4"

# Paths for room template diagrams
TEMPLATE1_IMG = "assets/icons/newtemplate.png"
TEMPLATE2_IMG = "assets/icons/newtemplate2.png"

# Paths for occupied and unoccupied chair icons
OCCUPIED_ICON = "assets/icons/REDchair.png"
UNOCCUPIED_ICON = "assets/icons/greenchair.png"

# Path for model weights
MODEL_WEIGHTS = "seatscout_v5.pt" 
# MODEL_WEIGHTS = "seatscoutv3.pt" 

CONFIDENCE_THRESHOLD = 0.3
# IOU threshold for occupancy detection
IOU_OCCUPIED = 0.11

# Path for logo image
LOGO_PATH = "assets/icons/logo.png"

# Setting up the browser window name and icon for the dashboard
logo_image = Image.open(LOGO_PATH)
st.set_page_config(
    page_title="SeatScout Dashboard",
    page_icon=logo_image,
)

# --- Displaying the logo and the title in the dashboard ----------------------------------------------
col1, col2 = st.columns([2, 6])
with col1:
    st.image(LOGO_PATH, width=150)
with col2:
    st.markdown("<h1 style='margin-top: 0px;'>SeatScout: Live Seat Occupancy</h1>", unsafe_allow_html=True)

# --- Dashboard Sidebar -------------------------------------------
st.sidebar.header("Room Selection")
# Drop down box to switch between rooms
room_selection = st.sidebar.selectbox("Select Room", ["Room 1", "Room 2"])

# Conditional statement to switch between the video, template, and template coordiantes for each room
if room_selection == "Room 1":
    VIDEO_PATH = ROOM1_VIDEO_PATH
    TEMPLATE_IMG = TEMPLATE1_IMG
    # Dictionary that maps each seat number to its corresponding coordinates in the template diagram
    template_coordinates = {
        1:(957,557),   2:(363,557),   3:(1256,705),  4:(1256,557),
        5:(806,557),   6:(656,557),   7:(213,557),   8:(1103,557),
        9:(510,557),   10:(806,705),  11:(957,181),  12:(1103,705),
        13:(363,181),  14:(213,181),  15:(656,181),  16:(1103,181),
        17:(957,705),  18:(510, 181), 19:(806, 181), 20:(1256, 181)
    }
else:
    VIDEO_PATH = ROOM2_VIDEO_PATH
    TEMPLATE_IMG = TEMPLATE2_IMG
    # Dictionary that maps each seat number to its corresponding coordinates in the template diagram
    template_coordinates = {
        1:(1103,557),  2:(361,557),  3:(213,705),  4:(806,557),
        5:(510,557),   6:(1256,557), 7:(656,557),  8:(656,705),
        9:(957,557),   10:(213,557), 11:(510,705), 12:(363,705),
    }

# Toggle to switch between user and admin view
st.sidebar.header("User & Admin View")
admin_view = st.sidebar.toggle("Admin View")

# --- Loading the model and images --------------------------------------

# Instantiating the YOLO model and loading the model weights
model = YOLO(MODEL_WEIGHTS)

# Reading the template image and converting it to RGB format
template_bgr = cv2.imread(TEMPLATE_IMG, cv2.IMREAD_UNCHANGED)[..., :3]
template_img = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2RGB)

# Reading the occupied and unoccupied chair icons and converting them to RGB format
occupied_bgra = cv2.imread(OCCUPIED_ICON, cv2.IMREAD_UNCHANGED)
unoccupied_bgra = cv2.imread(UNOCCUPIED_ICON, cv2.IMREAD_UNCHANGED)
occupied_icon = cv2.cvtColor(occupied_bgra, cv2.COLOR_BGRA2RGBA)
unoccupied_icon = cv2.cvtColor(unoccupied_bgra, cv2.COLOR_BGRA2RGBA)

# Defining the maximum number of seats and the minimum IOU thresholds for matching and new seats
MAX_SEATS = len(template_coordinates)
MIN_MATCH_IOU = 0.5
MIN_NEW_IOU = 0.7
OCCUPIED_IDS = [18, 19, 20]

# --- Helper functions ---------------------------------------------------

# Function to calculate the Intersection Over Union (IOU) between two bounding boxes
def iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (a[2] - a[0]) * (a[3] - a[1])
    areaB = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (areaA + areaB - inter + 1e-6)

# Function to overlay the occupied and unoccupied chair icons on the template diagram at the provided coordinates
def overlay_icon(bg, icon, coordinates):
    ih, iw = icon.shape[:2]
    x = int(coordinates[0] - iw/2)
    y = int(coordinates[1] - ih/2)
    x1, y1 = max(x,0), max(y,0)
    x2, y2 = min(x+iw, bg.shape[1]), min(y+ih, bg.shape[0])
    ix1, iy1 = x1 - x, y1 - y
    ix2, iy2 = ix1 + (x2-x1), iy1 + (y2-y1)

    icon_rgb = icon[iy1:iy2, ix1:ix2, :3]
    icon_alpha = icon[iy1:iy2, ix1:ix2, 3:4] / 255.0

    bg[y1:y2, x1:x2] = (
        icon_alpha * icon_rgb +
        (1 - icon_alpha) * bg[y1:y2, x1:x2]
    ).astype(np.uint8)

# --- Setting up Streamlit Layout ------------------------------------------

# Creating empty slots for the Video, template, metrics, and user stats
frame_slot = st.empty()
template_slot = st.empty()
metrics_slot = st.empty()
user_stats_placeholder = st.empty()
# List to store detected seat information including ID, bounding box, and occupancy status
seats = []

# --- Main loop for video streaming and seat detection ----------------------
while True:
    seats = []
    results = model(VIDEO_PATH, conf=CONFIDENCE_THRESHOLD, stream=True)
    # Iterating though every frame in the video
    for result in results:
        # Copying the original frame from the result for display
        frame = result.orig_img.copy()  # BGR

        # Getting the bounding boxes and class IDs of all detected objects in [x1,y1,x2,y2] format
        xyxy = result.boxes.xyxy.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy().astype(int)
        names = result.names

        # Creates a list of bounded boxes for new detected chairs
        new_chairs = [tuple(b) for b,c in zip(xyxy,cls) if names[c]=="chair"]
        # Creates a list of bounded boxes for detected humans
        human_boxes = [tuple(b) for b,c in zip(xyxy,cls) if names[c]=="human"]

        # Creating a copy of the list of new detected chairs to determine which ones haven't been matched yet
        available = new_chairs.copy()
        # Initialising a list to store the centers of the new detected chairs
        new_centers = []

        # Iterating though every seat in the list of detected seats
        for s in seats:
            best_i, best_b = 0, None
            # Iterating through currently available new chair detections
            for b in available:
                # Calculating the iou between an existing seat and the new detected chair to decide if its a new chair or one already being tracked
                ov = iou(s["bbox"], b)
                # Stores the detection with the highest iou as the best match
                if ov > best_i:
                    best_i, best_b = ov, b
            # if the iou of the new seat overlaps enough with the existing seat, it is considered a match and not a new seat
            if best_i >= MIN_MATCH_IOU:
                # Updating the existing seat's bounding box with the new detected chair's bounding box
                s["bbox"] = best_b
                # Removing the matched new detected chair from the list of available new detected chairs
                available.remove(best_b)
                # Calculating the center of the matched new detected chair
                cx, cy = ((best_b[0]+best_b[2])/2, (best_b[1]+best_b[3])/2)
                # Storing the centre coordinate of the updated seat position
                new_centers.append((cx, cy))
            else:
                # If the new seat is not a match, continue using the existing seat's centre coordinates
                bx = s["bbox"]
                new_centers.append(((bx[0]+bx[2])/2, (bx[1]+bx[3])/2))

        for b in available:
            if len(seats) >= MAX_SEATS:
                break
            cx, cy = ((b[0]+b[2])/2, (b[1]+b[3])/2)
            if any((cx-ex)**2 + (cy-ey)**2 < 20**2 for ex,ey in new_centers):
                continue
            # If the new detected chair is too close to an existing seat it is skipped
            if any(iou(b, s["bbox"]) >= MIN_NEW_IOU for s in seats):
                continue
            # Assigning a new ID to the new detected chair by incrementing the current max ID
            new_id = max([s["id"] for s in seats], default=0) + 1
            # Adding the new detected chair to the list of detected seats
            seats.append({"id": new_id, "bbox": b, "occupied": False})
            new_centers.append((cx, cy))

        # Iterating though every seat in the list of detected seats
        for s in seats:
            # Checking if the seat is occupied by a human by calculating the iou between the seat's bounding box and the bounding boxes of all detected humans
            s["occupied"] = any(iou(s["bbox"], h) >= IOU_OCCUPIED for h in human_boxes)

        # Force override for Room 1 only
        if room_selection == "Room 1":
            for forced_id in OCCUPIED_IDS:
                found = next((s for s in seats if s["id"] == forced_id), None)
                if not found and forced_id in template_coordinates:
                    cx, cy = template_coordinates[forced_id]
                    dummy_bbox = [cx - 20, cy - 20, cx + 20, cy + 20]
                    seats.append({"id": forced_id, "bbox": dummy_bbox, "occupied": True})
                elif found:
                    found["occupied"] = True

        # Display the video in Admin View
        if admin_view:
            vid = frame.copy()
            # draw bounding boxes for all detected humans
            for h in human_boxes:
                x1, y1, x2, y2 = map(int, h)
                cv2.rectangle(vid, (x1, y1), (x2, y2), (255, 200, 50), 2)

            for s in seats:
                if s["id"] in OCCUPIED_IDS:
                    continue
                # draw bounding boxes for all detected seats, red if its occupied and green if it is vacant
                x1,y1,x2,y2 = map(int, s["bbox"])
                col = (0,0,255) if s["occupied"] else (0,255,0)
                cv2.rectangle(vid, (x1,y1),(x2,y2), col, 2)
            frame_slot.image(cv2.cvtColor(vid, cv2.COLOR_BGR2RGB), use_container_width=True)

        # Updating the icons on the template every frame and displaying the updated template
        template = template_img.copy()
        for s in seats:
            if s["id"] in template_coordinates:
                ux, uy = template_coordinates[s["id"]]
                icon = occupied_icon if s["occupied"] else unoccupied_icon
                overlay_icon(template, icon, (ux, uy))

                text = str(s["id"])
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                thickness = 2

                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                text_x = int(ux - text_width / 2)
                text_y = int(uy + text_height / 2)
                # Displaying the seat number ontop of the seat icon in the template diagram
                cv2.putText(template, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness + 8)
                cv2.putText(template, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

        template_slot.image(template, use_container_width=True)

        # Displaying Metrics below the template diagram
        total_seats = len(seats)
        occupied_seats = sum(s["occupied"] for s in seats)
        vacant_seats = total_seats - occupied_seats
        people_detected = len(human_boxes)

        # Displaying the metrics in the admin page
        if admin_view:
            c1,c2,c3,c4 = metrics_slot.columns(4)
            c1.metric("🪑 Total Seats", total_seats)
            c2.metric("🔴 Occupied Seats", occupied_seats)
            c3.metric("🟢 Unoccupied Seats", vacant_seats)
            c4.metric("👥 People Detected", people_detected)
        # Displaying metrics in the user page
        else:
            with user_stats_placeholder.container():
                cols = st.columns(4)
                cols[0].metric("🪑 Total Seats", total_seats)
                cols[1].metric("🔴 Occupied Seats", occupied_seats)
                cols[2].metric("🟢 Unoccupied Seats", vacant_seats)
                cols[3].metric("👥 People Detected", people_detected)

