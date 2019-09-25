#!/usr/bin/env python

###############################################################################
# Copyright 2019 The Apollo Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

"""
This is a bunch of classes to manage cyber record channel extractor.
"""
import os
import sys
import struct
import numpy as np

import cv2
import pypcd

from modules.drivers.proto import sensor_image_pb2
from modules.drivers.proto import pointcloud_pb2
from modules.localization.proto import gps_pb2

from data_file_object import *

class SensorMessageParser(object):
    """Wrapper for cyber channel message extractor"""

    # Initalizing extractor
    def __init__(self, output_path, instance_saving=True):
        """
        instance_saving:
        True for large channel message, e.g., Camera/lidar/Radar;
        False for small channel message, e.g., GNSS topics
        """
        self._msg_parser = None
        self._timestamps = []
        self._proto_parser = None
        self._init_parser()
        self._parsed_data = None
        self._output_path = output_path
        self._timestamp_file = os.path.join(self._output_path, "timestamps.txt")
        self._instance_saving = instance_saving

    #initalizing msg and proto parser
    def _init_parser(self):
        raise NotImplementedError

    def parse_sensor_message(self, msg):
        raise NotImplementedError

    def save_messages_to_file(self):
        return True

    def get_msg_count(self):
        return len(self._timestamps)

    def get_timestamps(self):
        return self._timestamps

    def save_timestamps_to_file(self):
       timestamp_file_obj = TimestampFileObject(self._timestamp_file,
                                                operation='write',
                                                file_type='txt')
       timestamp_file_obj.save_to_file(self._timestamps)
       return True

class GpsParser(SensorMessageParser):
    """
    class to parse GNSS odometry channel.
    saving this small topic as a whole.
    """
    def __init__(self, output_path, instance_saving=False):
        super(GpsParser, self).__init__(output_path, instance_saving)
        if not self._instance_saving:
            self._parsed_data = []
            self._odomotry_output_file =\
             os.path.join(self._output_path, "Odometry.bin")

    def _init_parser(self):
        self._msg_parser = gps_pb2.Gps()

    def parse_sensor_message(self, msg):
        """ parse Gps information from GNSS odometry channel"""
        gps = self._msg_parser
        gps.ParseFromString(msg.message)

        # all double, except point_type is int32
        ts = gps.header.timestamp_sec
        self._timestamps.append(ts)

        point_type = 0
        qw = gps.localization.orientation.qw
        qx = gps.localization.orientation.qx
        qy = gps.localization.orientation.qy
        qz = gps.localization.orientation.qz
        x = gps.localization.position.x
        y = gps.localization.position.y
        z = gps.localization.position.z
        # save 9 values as a tuple, for eaisier struct packing during storage
        if self._instance_saving:
            raise ValueError("Gps odometry should be saved in a file")
        else:
            self._parsed_data.append((ts, point_type, qw, qx, qy, qz, x, y, z))

        return True

    def save_messages_to_file(self):
        """save list of parsed Odometry messages to file"""
        odometry_file_obj = OdometryFileObject(file_path=self._odomotry_output_file,
                                               operation='write',
                                               file_type='binary')
        odometry_file_obj.save_to_file(self._parsed_data)
        return True

class PointCloudParser(SensorMessageParser):
    """
    class to parse apollo/$(lidar)/PointCloud2 channels.
    saving seperately each parsed msg
    """
    def __init__(self, output_path, instance_saving=True):
        super(PointCloudParser, self).__init__(output_path, instance_saving)
        self._save_csv = True
        self._save_pcd = False

    def convert_xyzit_pb_to_array(self, xyz_i_t, data_type, is_dense=True, laser_cnt=0):
        arr = np.zeros(len(xyz_i_t), dtype=data_type)
        for i, point in enumerate(xyz_i_t):
            # change timestamp to timestamp_sec
            if is_dense:
                arr[i] = (point.x, point.y, point.z,
                          point.intensity, point.timestamp/1e9)
            else:
                arr[i] = (point.x, point.y, point.z, point.intensity, i % laser_cnt)
        return arr

    def make_xyzit_point_cloud(self, xyz_i_t):
        """
        Make a pointcloud object from PointXYZIT message, as Pointcloud.proto.
        message PointXYZIT {
          optional float x = 1 [default = nan];
          optional float y = 2 [default = nan];
          optional float z = 3 [default = nan];
          optional uint32 intensity = 4 [default = 0];
          optional uint64 timestamp = 5 [default = 0];
        }
        """

        md = {'version': .7,
              'fields': ['x', 'y', 'z', 'intensity', 'timestamp'],
              'count': [1, 1, 1, 1, 1],
              'width': len(xyz_i_t),
              'height': 1,
              'viewpoint': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
              'points': len(xyz_i_t),
              'type': ['F', 'F', 'F', 'U', 'F'],
              'size': [4, 4, 4, 4, 8],
              'data': 'binary_compressed'}

        typenames = []
        for t, s in zip(md['type'], md['size']):
            np_type = pypcd.pcd_type_to_numpy_type[(t, s)]
            typenames.append(np_type)

        np_dtype = np.dtype(zip(md['fields'], typenames))
        pc_data = self.convert_xyzit_pb_to_array(xyz_i_t, data_type=np_dtype)
        pc = pypcd.PointCloud(md, pc_data)
        return pc

    def make_xyzir_point_cloud(self, xyz_i_t, laser_cnt):
        """
        Make a pointcloud object from PointXYZIT message, as Pointcloud.proto.
        message PointXYZIT {
          optional float x = 1 [default = nan];
          optional float y = 2 [default = nan];
          optional float z = 3 [default = nan];
          optional uint32 intensity = 4 [default = 0];
          optional uint64 timestamp = 5 [default = 0];
        }
        """
        print('len xyzit is %d' % len(xyz_i_t))
        print('len per laser is %d' % (len(xyz_i_t) / 64))
        md = {'version': .7,
              'fields': ['x', 'y', 'z', 'intensity', 'ring'],
              'count': [1, 1, 1, 1, 1],
              'width': len(xyz_i_t),
              'height': 1,
              'viewpoint': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
              'points': len(xyz_i_t),
              'type': ['F', 'F', 'F', 'F', 'U'],
              'size': [4, 4, 4, 4, 2],
              'data': 'binary_compressed'}

        typenames = []
        for t, s in zip(md['type'], md['size']):
            np_type = pypcd.pcd_type_to_numpy_type[(t, s)]
            typenames.append(np_type)

        np_dtype = np.dtype(zip(md['fields'], typenames))
        pc_data = self.convert_xyzit_pb_to_array(xyz_i_t, data_type=np_dtype,
                                                 is_dense=False, laser_cnt=laser_cnt)
        pc = pypcd.PointCloud(md, pc_data)
        return pc

    def save_pointcloud_meta_to_file(self, pc_meta, pcd_file):
        pypcd.save_point_cloud_bin_compressed(pc_meta, pcd_file)

    def save_pointcloud_to_csv(self, pc_meta, csv_file):
        pypcd.save_txt(pc_meta, csv_file)

    def _init_parser(self):
        self._msg_parser = pointcloud_pb2.PointCloud()

    def parse_sensor_message(self, msg):
        """
        Transform protobuf PointXYZIT to standard PCL bin_compressed_file(*.pcd).
        """
        pointcloud = self._msg_parser
        pointcloud.ParseFromString(msg.message)

        self._timestamps.append(pointcloud.measurement_time)
        # self._timestamps.append(pointcloud.header.timestamp_sec)
        if pointcloud.is_dense:
            print('process dense point cloud')
            self._parsed_data = self.make_xyzit_point_cloud(pointcloud.point)
        else:
            print('process not dense point cloud')
            #TODO: make laser cnt configurable
            self._parsed_data = self.make_xyzir_point_cloud(pointcloud.point, 64)

        if self._instance_saving:
            if self._save_pcd:
                file_name = "%06d.pcd" % self.get_msg_count()
                output_file = os.path.join(self._output_path, file_name)
                self.save_pointcloud_meta_to_file(pc_meta=self._parsed_data, pcd_file=output_file)
            if self._save_csv:
                file_name = "%06d.csv" % self.get_msg_count()
                output_file = os.path.join(self._output_path, file_name)
                self.save_pointcloud_to_csv(pc_meta=self._parsed_data, csv_file=output_file)
        else:
            raise ValueError("not implement multiple message concatenation for PointCloud2 topic")
        # TODO(gchen-Apollo): add saint check
        return True


class ImageParser(SensorMessageParser):
    """
    class to parse apollo/$(camera)/image channels.
    saving seperately each parsed msg
    """
    def __init__(self, output_path, instance_saving=True):
        super(ImageParser, self).__init__(output_path, instance_saving)

    def _init_parser(self):
        self._msg_parser = sensor_image_pb2.Image()

    def parse_sensor_message(self, msg):

        image = self._msg_parser
        image.ParseFromString(msg.message)

        self._timestamps.append(image.header.timestamp_sec)
        # Save image according to cyber format, defined in sensor camera proto.
        # height = 4, image height, that is, number of rows.
        # width = 5,  image width, that is, number of columns.
        # encoding = 6, as string, type is 'rgb8', 'bgr8' or 'gray'.
        # step = 7, full row length in bytes.
        # data = 8, actual matrix data in bytes, size is (step * rows).
        # type = CV_8UC1 if image step is equal to width as gray, CV_8UC3
        # if step * 3 is equal to width.
        if image.encoding == 'rgb8' or image.encoding == 'bgr8':
            if image.step != image.width * 3:
                print('Image.step %d does not equal to Image.width %d * 3 for color image.'
                      % (image.step, image.width))
                return False
        elif image.encoding == 'gray' or image.encoding == 'y':
            if image.step != image.width:
                print('Image.step %d does not equal to Image.width %d or gray image.'
                      % (image.step, image.width))
                return False
        else:
            print('Unsupported image encoding type: %s.' % image.encoding)
            return False

        channel_num = image.step / image.width
        self._parsed_data = np.fromstring(image.data, dtype=np.uint8).reshape(
            (image.height, image.width, channel_num))

        if self._instance_saving:
            file_name = "%06d.png" % self.get_msg_count()
            output_file = os.path.join(self._output_path, file_name)
            self.save_image_mat_to_file(image_file=output_file)
        else:
            raise ValueError("not implement multiple message concatenation for Image topic")

        return True

    def save_image_mat_to_file(self, image_file):
        # Save image in BGR oder
        image_mat = self._parsed_data
        if self._msg_parser.encoding == 'rgb8':
            cv2.imwrite(image_file, cv2.cvtColor(image_mat, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(image_file, image_mat)
