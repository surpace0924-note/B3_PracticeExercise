#!/bin/bash

#sudo python web_control_server2.py&

cd /usr/src/mjpeg-streamer/mjpg-streamer/mjpg-streamer-experimental
export LD_LIBRARY_PATH=.
./mjpg_streamer -o "output_http.so -w ./www -p 8080" -i "input_raspicam.so -x 100 -y 160 -q 8 -fps 8 quality 7"

