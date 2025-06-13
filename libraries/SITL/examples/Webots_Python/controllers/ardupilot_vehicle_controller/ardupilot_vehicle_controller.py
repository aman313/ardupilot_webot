'''
General ardupilot vehicle controller for Webots 2023a

AP_FLAKE8_CLEAN
'''


import time
import argparse
from webots_vehicle import WebotsArduVehicle
import sys
from pymavlink import mavutil
from PIL import Image
import numpy as np

if sys.version_info.major == 3 and sys.version_info.minor >= 10:
    import collections
    setattr(collections, "MutableMapping", collections.abc.MutableMapping)
    
from dronekit import connect, VehicleMode
import math


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--motors", "-m",
                        type=str,
                        default="m1_motor, m2_motor, m3_motor, m4_motor",
                        help="Comma spaced list of motor names in ardupilot numerical order (ex --motors \"m1,m2,m3, m4\")")
    parser.add_argument("--reversed-motors", "-r",
                        type=str,
                        default=None,
                        help="Comma spaced list of motors to reverse (starting from 1, in ardupilot order)")
    parser.add_argument("--bidirectional-motors",
                        type=bool,
                        default=False,
                        help="If the motors are bidirectional (as is the case for Rovers usually)")
    parser.add_argument("--uses-propellers",
                        type=bool,
                        default=True,
                        help="Whether the vehicle uses propellers. This is important as we need to linearize thrust if so")
    parser.add_argument("--motor-cap",
                        type=float,
                        default=float('inf'),
                        help="Motor velocity cap. This is useful for the crazyflie which default has way too much power")

    parser.add_argument("--accel",
                        type=str,
                        default="accelerometer",
                        help="Webots accelerometer name")
    parser.add_argument("--imu",
                        type=str,
                        default="inertial unit",
                        help="Webots IMU name")
    parser.add_argument("--gyro",
                        type=str,
                        default="gyro",
                        help="Webots gyro name")
    parser.add_argument("--gps",
                        type=str,
                        default="gps",
                        help="Webots GPS name")

    parser.add_argument("--camera",
                        type=str,
                        default=None,
                        help="Webots Camera name (optional)")
    parser.add_argument("--camera-fps",
                        type=int,
                        default=10,
                        help="Camera FPS. Note lower FPS is faster")
    parser.add_argument("--camera-port",
                        type=int,
                        default=None,
                        help="Port to stream grayscale camera images to. "
                             "If no port is supplied the camera will not be streamed.")

    parser.add_argument("--rangefinder",
                        type=str,
                        default=None,
                        help="Webots RangeFinder name (optional)")
    parser.add_argument("--rangefinder-fps",
                        type=int,
                        default=10,
                        help="rangefinder FPS. Note lower FPS is faster")
    parser.add_argument("--rangefinder-port",
                        type=int,
                        default=None,
                        help="Port to stream grayscale rangefinder images to. "
                             "If no port is supplied the rangefinder will not be streamed.")

    parser.add_argument("--instance", "-i",
                        type=int,
                        default=0,
                        help="Drone instance to match the SITL. This allows multiple vehicles")
    parser.add_argument("--sitl-address",
                        type=str,
                        default="127.0.0.1",
                        help="IP address of the SITL (useful with WSL2 eg \"172.24.220.98\")")

    return parser.parse_args()


from dronekit import Vehicle
from pymavlink import mavutil
import math, time

# ────────────────────────────────────────────────
# Helper: Euler → quaternion  (degrees in, list[w,x,y,z] out)
def to_quaternion(roll=0.0, pitch=0.0, yaw=0.0):
    """Convert roll, pitch, yaw (deg) to quaternion suitable for MAVLink."""
    r  = math.radians(roll)
    p  = math.radians(pitch)
    y  = math.radians(yaw)
    cy, sy = math.cos(y*0.5), math.sin(y*0.5)
    cr, sr = math.cos(r*0.5), math.sin(r*0.5)
    cp, sp = math.cos(p*0.5), math.sin(p*0.5)
    return [
        cy*cr*cp + sy*sr*sp,      # w
        cy*sr*cp - sy*cr*sp,      # x
        cy*cr*sp + sy*sr*cp,      # y
        sy*cr*cp - cy*sr*sp       # z
    ]

# ────────────────────────────────────────────────
def yaw_relative(vehicle: Vehicle, offset_deg: float,
                 duration=2.0, thrust=0.5, rate_hz=10):
    """
    Yaw by `offset_deg` relative to current heading using SET_ATTITUDE_TARGET.
    `duration` is how long to keep resending the set-point (FCU requires ≥ 1 s).
    """
    # 1.  Grab present yaw (DroneKit gives radians in [-π,π])
    current_yaw = math.degrees(vehicle.attitude.yaw)
    target_yaw  = (current_yaw + offset_deg) % 360

    # 2.  Build quaternion for level attitude + new yaw
    q = to_quaternion(0, 0, target_yaw)

    # 3.  Type-mask:  bit 1‒3 = ignore body-rates.
    #                 everything else (angles & thrust) honoured.
    TYPE_MASK = 0b00000111

    # 4.  Stream the message for `duration`
    t0 = time.time()
    while time.time() - t0 < duration:
        msg = vehicle.message_factory.set_attitude_target_encode(
            0,          # time_boot_ms (not used)
            0, 0,       # target system, component
            TYPE_MASK,  # ignore body rates, use quaternion + thrust
            q,          # attitude quaternion w,x,y,z
            0, 0, 0,    # body roll/pitch/yaw rates (ignored because mask)
            thrust      # 0–1 (≈0.5 holds altitude in AltHold/Guided)
        )
        vehicle.send_mavlink(msg)
        time.sleep(1.0/rate_hz)

    # 5.  OPTIONAL: clear the stick-like set-point so the vehicle
    #     goes back to flight-controller’s normal yaw stabilisation
    vehicle.flush()          # pushes any buffered MAVLink immediately


def wait_for_yaw(vehicle, target_yaw, timeout=10, tolerance=0.1):
    """
    Wait for the vehicle to reach the target yaw angle
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        current_yaw = vehicle.attitude.yaw
        if abs(current_yaw - target_yaw) < tolerance:
            return True
        time.sleep(0.1)
    return False


if __name__ == "__main__":
    args = get_args()

    # parse string arguments into lists
    motors = [x.strip() for x in args.motors.split(',')]
    if args.reversed_motors:
        reversed_motors = [int(x) for x in args.reversed_motors.split(",")]
    else:
        reversed_motors = []

    vehicle = WebotsArduVehicle(motor_names=motors,
                                reversed_motors=reversed_motors,
                                accel_name=args.accel,
                                imu_name=args.imu,
                                gyro_name=args.gyro,
                                gps_name=args.gps,
                                camera_name=args.camera,
                                camera_fps=args.camera_fps,
                                camera_stream_port=args.camera_port,
                                rangefinder_name=args.rangefinder,
                                rangefinder_fps=args.rangefinder_fps,
                                rangefinder_stream_port=args.rangefinder_port,
                                instance=args.instance,
                                motor_velocity_cap=args.motor_cap,
                                bidirectional_motors=args.bidirectional_motors,
                                uses_propellers=args.uses_propellers,
                                sitl_address=args.sitl_address)

    # Connect to the drone
    connection_string = f"tcp:{args.sitl_address}:5760"
    drone = connect(connection_string, wait_ready=True)
    
    # Wait for the drone to be ready
    while not drone.is_armable:
        print("Waiting for drone to be armable...")
        time.sleep(1)
    
    # Arm the drone
    print("Arming motors...")
    drone.mode = VehicleMode("GUIDED")
    drone.armed = True
    
    # Wait for arming
    while not drone.armed:
        print("Waiting for arming...")
        time.sleep(1)
    
    # Take off to target altitude
    target_altitude = 5  # meters
    print(f"Taking off to {target_altitude} meters...")
    drone.simple_takeoff(target_altitude)
    
    # Wait until we reach target altitude
    while True:
        current_altitude = drone.location.global_relative_frame.alt
        if current_altitude >= target_altitude * 0.95:  # Within 5% of target
            print("Reached target altitude")
            break
        time.sleep(1)

    # Rotate and capture images
    print("Starting rotation and image capture sequence...")
    for i in range(4):  # 4 positions (original + 3 rotations)
        offset_deg = 90*i
        drone.mode = VehicleMode("GUIDED")      # or "GUIDED_NOGPS"
        yaw_relative(drone, offset_deg=90) 
        # turn 90 deg CW
        
        # # Wait for yaw to reach target
        # if wait_for_yaw(drone, offset_deg):
        #     print(f"Reached target yaw of {math.degrees(offset_deg)} degrees")
        # else:
        #     print(f"Failed to reach target yaw of {math.degrees(target_yaw)} degrees")
        #     continue
        
        time.sleep(4)  # Additional stabilization time
        
        # Capture and save image
        if args.camera:
            print(f"Capturing image at {offset_deg} degrees...")
            image = vehicle.get_camera_image()
            if image is not None:
                filename = f"/Users/aman/Documents/capture_{offset_deg}_degrees.jpg"
                # Convert numpy array to PIL Image and save
                pil_image = Image.fromarray(image)
                pil_image.save(filename)
                print(f"Saved image to {filename}")
            else:
                print("Failed to capture image")
        else:
            print("No camera found")
        
        time.sleep(1)  # Wait before next rotation

    while vehicle.webots_connected():
        time.sleep(1)
