# Ball Tracking with Live Trajectory Visualization

![Python](https://img.shields.io/badge/python-3.13-blue.svg)

This project demonstrates real-time ball tracking with live trajectory visualization using OpenCV and Matplotlib. The system tracks a ball thrown vertically into the air and analyzes its motion by plotting position, velocity, and acceleration in real-time. Polynomial functions are used to model the ball's motion and predict its trajectory based on physics principles.

![Ball Tracking Demo](https://github.com/MAINAKSAHA07/ball-tracking-live-plot/assets/demo.gif)

## 🌟 Features

- **Real-time ball tracking** using OpenCV computer vision
- **Live trajectory visualization** with Matplotlib
- **Physics-based prediction** using polynomial fitting
- **Performance optimization** through matplotlib blitting
- **Streaming trajectory display** with fixed-length buffer
- **Multiple visualization modes** for different use cases
- **Advanced visualization with Rerun SDK** for interactive 3D plots
- **Video looping and processing utilities**
- **Color-based and motion-based ball detection**
- **Trajectory smoothing and alpha blending**
- **Video saving capabilities**
- **🌐 Web Application** for real-time video upload and analysis

## 🚀 Quick Start

This project uses [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management. Simply run the scripts with uv to get started!

### Real-time Tracking with Streaming Visualization
```bash
uv run tracking
```

### Trajectory Analysis with Physics Prediction
```bash
uv run trajectory
```

### Advanced Visualization with Rerun SDK
```bash
uv run python src/ball_tracking/trajectory_rerun.py
```

### 🌐 Web Application
```bash
uv run web
```
Then open http://localhost:5000 in your browser to upload and analyze videos!

## 🌐 Web Application Features

The web application provides a modern, user-friendly interface for ball tracking analysis:

### ✨ Key Features
- **Drag & Drop Upload**: Easy video file upload with drag-and-drop support
- **Real-time Analysis**: Process videos frame-by-frame with customizable parameters
- **Live Visualization**: See the ball tracking in real-time with trajectory overlay
- **Interactive Controls**: Adjust start/end frames and processing step size
- **Statistics Dashboard**: View detailed analysis statistics and confidence metrics
- **Responsive Design**: Works on desktop and mobile devices
- **Session Management**: Automatic cleanup of uploaded files and analysis sessions

### 🎯 Supported Video Formats
- MP4, AVI, MOV, MKV, WEBM
- Maximum file size: 100MB
- Any resolution supported

### 🔧 Analysis Parameters
- **Start Frame**: Begin analysis from a specific frame
- **End Frame**: Stop analysis at a specific frame
- **Step Size**: Process every Nth frame for faster analysis

### 📊 Analysis Results
- **Frame-by-frame tracking**: See each processed frame with ball detection
- **Trajectory visualization**: Interactive canvas showing ball path
- **Statistics**: Frames processed, ball detections, confidence scores
- **Real-time progress**: Live progress bar during analysis

## 📦 Installation

### Prerequisites
- Python 3.13 or higher
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/MAINAKSAHA07/ball-tracking-live-plot.git
   cd ball-tracking-live-plot-main
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

3. Run the application:
   ```bash
   uv run tracking  # For real-time tracking
   uv run trajectory  # For trajectory analysis
   uv run python src/ball_tracking/trajectory_rerun.py  # For Rerun visualization
   uv run web  # For web application
   ```

## 🚀 Deployment

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run the web application
python -m ball_tracking.web
```

### Production Deployment

1. **Setup the application**:
   ```bash
   python deploy.py
   ```

2. **Using Gunicorn (recommended for production)**:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 ball_tracking.web:app
   ```

3. **Environment Variables**:
   ```bash
   export FLASK_ENV=production
   export FLASK_DEBUG=0
   export MAX_CONTENT_LENGTH=104857600
   ```

4. **Reverse Proxy Setup (Nginx example)**:
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;
       
       location / {
           proxy_pass http://127.0.0.1:5000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   ```

### Docker Deployment
```dockerfile
FROM python:3.13-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN mkdir -p uploads results logs

EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "ball_tracking.web:app"]
```

## 📁 Project Structure

```
src/ball_tracking/
├── __init__.py
├── core.py              # Core data structures (Point2D type)
├── tracking.py          # Real-time ball tracking with streaming visualization
├── trajectory.py        # Trajectory analysis with physics prediction
├── trajectory_rerun.py  # Advanced visualization using Rerun SDK
├── video_loop.py        # Video processing utilities with looping support
├── colormap.py          # Color mapping utilities for trajectory visualization
├── web.py              # 🌐 Web application with Flask
└── templates/
    └── index.html      # Modern web interface

media/                   # Sample video files
├── ball.mp4
├── ball2.mp4
├── ball3.mp4
├── ball4.mp4
└── kick1.MOV

uploads/                 # Uploaded video files (created automatically)
results/                 # Analysis results (created automatically)
requirements.txt         # Python dependencies
deploy.py               # Deployment script
```

## 📈 Technical Implementation

### Ball Detection Algorithm

The system uses a two-stage detection approach:

1. **Color-based filtering**: HSV color space filtering to detect green/yellow balls
2. **Motion-based filtering**: Background subtraction using MOG2 to detect moving objects
3. **Contour analysis**: Finding the largest contour to identify the ball
4. **Trajectory smoothing**: Alpha blending for smooth position tracking

```python
# Color filtering in HSV space
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask_color = cv2.inRange(hsv, np.array([30, 30, 30]), np.array([100, 150, 150]))

# Motion detection with background subtraction
bg_sub = cv2.createBackgroundSubtractorMOG2(varThreshold=128, detectShadows=False)
mask_fg = bg_sub.apply(frame, learningRate=0)

# Combine masks and find contours
mask = cv2.bitwise_and(mask_color, mask_fg)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

### Matplotlib + OpenCV Integration

To integrate matplotlib plots into OpenCV for real-time display, the canvas is rendered into a memory buffer, converted to a numpy array, and transformed to the correct BGR format:

```python
fig.canvas.draw()

buf = fig.canvas.buffer_rgba()
plot = np.asarray(buf)
plot = cv2.cvtColor(plot, cv2.COLOR_RGB2BGR)
```

### 🎨 Performance Optimization with Blitting

The naive approach of redrawing the entire matplotlib figure every frame is computationally expensive. To improve performance, the project uses [blitting](https://matplotlib.org/stable/users/explain/animations/blitting.html) - a technique that only redraws the regions that have changed, drastically reducing rendering time.

```python
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 2), dpi=100)

# Initialize empty plots
pl_pos = axs[0].plot([], [], c="b")[0]
pl_vel = axs[1].plot([], [], c="b")[0]
pl_acc = axs[2].plot([], [], c="b")[0]

# Draw initial backgrounds
fig.canvas.draw()
bg_axs = [fig.canvas.copy_from_bbox(ax.bbox) for ax in axs]

while True:
    # Update plot data
    pl_pos.set_data(range(len(pos)), pos)
    pl_vel.set_data(range(len(vel)), vel)
    pl_acc.set_data(range(len(acc)), acc)
    
    # Blit each subplot for optimal performance
    for i, (ax, pl, bg) in enumerate(zip(axs, [pl_pos, pl_vel, pl_acc], bg_axs)):
        fig.canvas.restore_region(bg)
        ax.draw_artist(pl)
        fig.canvas.blit(ax.bbox)
```

### 🔮 Physics-Based Trajectory Prediction

The trajectory prediction is based on a physics model assuming constant acceleration equal to gravitational acceleration (g ≈ 9.81 m/s²). The ball's motion can be described by polynomial functions:

- **Position**: 2nd degree polynomial (quadratic)
- **Velocity**: 1st degree polynomial (linear) 
- **Acceleration**: 0th degree polynomial (constant)

The polynomial coefficients are calculated using numpy's `polyfit` function, and the predictions are evaluated and plotted alongside the tracked motion:

```python
# Fit polynomials to the tracked data
poly_pos = np.polyfit(t_pos, pos, deg=2)
poly_vel = np.polyfit(t_vel, vel, deg=1)
poly_acc = np.polyfit(t_acc, acc, deg=0)

# Generate predictions for future time steps
t_pred = np.arange(num_frames + 5)
polyval_pos = np.polyval(poly_pos, t_pred)
polyval_vel = np.polyval(poly_vel, t_pred)
polyval_acc = np.polyval(poly_acc, t_pred)
```

### 🎥 Streaming Trajectory Visualization

The real-time tracking mode uses Python's `collections.deque` to maintain a fixed-length buffer of trajectory points, creating a smooth streaming effect that shows the ball's recent path without overwhelming the display.

### 🌈 Rainbow Colormap for Trajectory

The trajectory is visualized using a rainbow colormap that shows the temporal progression of the ball's path:

```python
def colormap_rainbow(value: float) -> tuple[int, int, int]:
    """Map a value between 0 and 1 to a color in the rainbow colormap."""
    pixel_img = np.array([[value * 255]], dtype=np.uint8)
    pixel_cmap_img = cv2.applyColorMap(pixel_img, cv2.COLORMAP_RAINBOW)
    return pixel_cmap_img.flatten().tolist()
```

### 🎬 Video Loop Utility

The `VideoLoop` class provides a context manager for video processing with features like:
- Automatic frame timing control
- Video looping support
- Skip seconds functionality
- Proper resource management

### 🚀 Rerun SDK Integration

The `trajectory_rerun.py` script provides an alternative visualization using the Rerun SDK, offering:
- Interactive 3D visualization
- Time series plots for position, velocity, and acceleration
- Real-time data streaming
- Advanced layout customization

## 🎯 Use Cases

- **Computer Vision Education**: Learn real-time object tracking techniques
- **Physics Visualization**: Understand projectile motion and kinematics
- **Performance Optimization**: Study matplotlib blitting for real-time plotting
- **Sports Analysis**: Analyze ball trajectories in sports applications
- **Prototype Development**: Base for more complex tracking systems
- **Data Visualization**: Explore different visualization frameworks (Matplotlib vs Rerun)

## 🛠️ Dependencies

- **OpenCV** (`opencv-python`): Computer vision and video processing
- **Matplotlib**: Real-time plotting and visualization
- **NumPy**: Numerical computations and array operations
- **Rerun SDK**: Advanced visualization framework for interactive plots

## 📝 License

This project is open source. Please check the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Feel free to submit issues and pull requests to improve the project.
