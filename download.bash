mkdir kitti;
cd kitti;
wget -c http://kitti.is.tue.mpg.de/kitti/data_object_calib.zip &
wget -c http://kitti.is.tue.mpg.de/kitti/data_object_label_2.zip &
wget -c http://kitti.is.tue.mpg.de/kitti/data_object_velodyne.zip ;

unzip data_object_calib.zip &
unzip data_object_label_2 &
unzip data_object_velodyne ;

echo "Unzipped and ready"
cd ..




