sudo chmod 777 /dev/ttyACM0 & sleep 3;
gnome-terminal -- bash -c "roslaunch mavros px4.launch; exec bash" & sleep 5
taskset -c 0-7 roslaunch vins realsense_d435i.launch & sleep 5;
roslaunch realsense2_camera rs_camera.launch & sleep 10;
roslaunch vins vins_rviz.launch & sleep 5;
rosrun mavros mavcmd long 511 105 5000 0 0 0 0 0 & sleep 2;
rosrun mavros mavcmd long 511 31 5000 0 0 0 0 0 & sleep 2;
gnome-terminal -- bash -c "roslaunch vins_to_mavros vins_to_mavros.launch; exec bash" & sleep 5
gnome-terminal -- bash -c "roslaunch ego_planner single_run_in_exp.launch; exec bash" & sleep 5
roslaunch ego_planner rviz.launch & sleep 5;
wait;
