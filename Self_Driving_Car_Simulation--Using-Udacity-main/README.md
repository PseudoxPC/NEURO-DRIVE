# Self-Driving Car Simulation using Udacity Simulator

This project demonstrates a basic self-driving car pipeline using behavioral cloning. It trains a convolutional neural network (CNN) to predict steering angles based on front-facing camera images, and then tests the trained model using the Udacity Self-Driving Car Simulator.

---

## Project Structure

```
.
├── TrainingSimulation.py       # Main training script
├── TestSimulation.py           # Model testing & inference server
├── utils.py                    # Helper functions for data handling, augmentation, and model definition
├── myData/                     # Directory containing driving data (images and driving_log.csv)
│   └── driving_log.csv         # CSV file with image paths and driving metrics
│   └── IMG/                    # Folder containing captured images
├── model1.h5                   # Trained Keras model (generated after training)
```

---

## Features

- Data preprocessing (cropping, resizing, color space conversion, normalization)
- Data balancing to avoid bias towards straight driving
- Data augmentation (panning, zoom, brightness adjustment, flipping)
- CNN architecture inspired by NVIDIA’s end-to-end self-driving car model
- Real-time inference using Flask-SocketIO server with the simulator

---

## Dependencies

Ensure you have the following libraries installed:

```bash
pip install numpy pandas opencv-python matplotlib scikit-learn imgaug flask eventlet tensorflow
```

---

## Model Training

To train the model:

1. Record driving data using the simulator.
2. Save the data into the `myData/` directory.
3. Run the training script:

```bash
python TrainingSimulation.py
```

This will:
- Import and preprocess the data
- Balance and augment the dataset
- Train the model using NVIDIA’s architecture
- Save the trained model as `model1.h5`

---

## Model Testing & Simulation

To test the trained model with the simulator:

1. Open the Udacity Self-Driving Car Simulator.
2. Select **Autonomous Mode**.
3. Run the testing server:

```bash
python TestSimulation.py
```

This launches a Flask-based server that:
- Receives live images from the simulator
- Preprocesses and feeds them into the trained CNN
- Sends back steering and throttle commands in real-time

---

## Model Architecture

Defined in `utils.py` (`createModel()` function):

- 5 convolutional layers with ELU activation
- Flatten + Fully connected layers
- Final output: steering angle
- Optimizer: Adam (lr = 0.0001)
- Loss function: Mean Squared Error (MSE)

---

## Visualization

During training, training and validation loss curves are plotted for each epoch to monitor overfitting and performance.

---

## Notes

- Data augmentation is only applied to training data, not validation.
- Model expects preprocessed image input of shape `(66, 200, 3)` in YUV color space.
- Speed throttle is dynamically adjusted based on current speed to mimic realistic driving.

---

## Credits

This project is based on the behavioral cloning concept from the Udacity Self-Driving Car Nanodegree and NVIDIA's end-to-end deep learning research.
