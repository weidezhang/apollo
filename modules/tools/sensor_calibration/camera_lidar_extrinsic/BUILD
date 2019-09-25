load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "feature_extractor",
    hdrs = [
        "point_type.h",
        "feature_extractor.h",
        "calibration_data.h",
        "config.h",
    ],
    srcs = [
        "feature_extractor.cc",
    ],
    linkopts = [
        "-lopencv_core",
        "-lopencv_imgproc",
        "-lopencv_calib3d",
        "-lopencv_highgui",
    ],
    deps = [
        "@pcl",
        "@vtk",
        "@eigen",
    ],
)

cc_binary(
    name = "lidar_cam_calibrate",
    srcs = ["main.cc"],
    deps = [
        "//cyber",
        "//modules/tools/sensor_calibration/camera_lidar_extrinsic:feature_extractor",
        "//modules/common/util",
    ],
)



cpplint()