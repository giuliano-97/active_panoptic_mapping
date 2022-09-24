#!/usr/bin/env python3

"""
The source code in this file is based on:
https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/reader.py

Copyright 2017 
Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, 
Matthias Niessner

Permission is hereby granted, free of charge, to any person obtaining a copy of 
this software and associated documentation files (the "Software"), to deal in 
the Software without restriction, including without limitation the rights to 
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
of the Software, and to permit persons to whom the Software is furnished to do 
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
"""

import argparse
import os
import sys
import struct
import csv
from typing import Tuple

import numpy as np
import zlib
import imageio
import cv2
import png

COMPRESSION_TYPE_COLOR = {-1: "unknown", 0: "raw", 1: "png", 2: "jpeg"}
COMPRESSION_TYPE_DEPTH = {
    -1: "unknown",
    0: "raw_ushort",
    1: "zlib_ushort",
    2: "occi_ushort",
}


# params
parser = argparse.ArgumentParser()
# data paths
parser.add_argument(
    "--filename",
    required=True,
    help="path to sens file to read",
)
parser.add_argument(
    "--output_path",
    help="path to output folder",
)
parser.add_argument(
    "--export_depth",
    dest="export_depth",
    action="store_true",
)
parser.add_argument(
    "--export_color",
    dest="export_color",
    action="store_true",
)
parser.add_argument(
    "--export_pose",
    dest="export_pose",
    action="store_true",
)
parser.add_argument(
    "--export_intrinsic",
    dest="export_intrinsic",
    action="store_true",
)
parser.add_argument(
    "--export_timestamps",
    dest="export_timestamps",
    action="store_true",
)
parser.add_argument(
    "--image_size",
    dest="image_size",
    type=int,
    nargs=2,
    help="Size of the exported color images as WIDTH HEIGHT",
)
parser.set_defaults(
    output_path=None,
    export_depth=False,
    export_color=False,
    export_pose=False,
    export_intrinsic=False,
    export_timestamps=False,
    image_size=None,
)

opt = parser.parse_args()
print(opt)


def format_frame_number_with_leading_zeros(frame: int) -> str:
    """
    Formats the given number as a 5-digit with leading zeros
    """
    return "{:05d}".format(frame)


def adjust_intrinsic_matrix(
    intrinsic_matrix: np.ndarray,
    original_image_size: Tuple[int, int],
    new_image_size: Tuple[int, int],
):
    rx = new_image_size[0] / original_image_size[0]
    ry = new_image_size[1] / original_image_size[1]
    S = np.diag([rx, ry, 1, 1])
    return S.dot(intrinsic_matrix)


class RGBDFrame:
    def load(self, file_handle):
        self.camera_to_world = np.asarray(
            struct.unpack("f" * 16, file_handle.read(16 * 4)), dtype=np.float32
        ).reshape(4, 4)
        self.timestamp_color = struct.unpack("Q", file_handle.read(8))[0]
        self.timestamp_depth = struct.unpack("Q", file_handle.read(8))[0]
        self.color_size_bytes = struct.unpack("Q", file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack("Q", file_handle.read(8))[0]
        self.color_data = b"".join(
            struct.unpack(
                "c" * self.color_size_bytes, file_handle.read(self.color_size_bytes)
            )
        )
        self.depth_data = b"".join(
            struct.unpack(
                "c" * self.depth_size_bytes, file_handle.read(self.depth_size_bytes)
            )
        )

    def decompress_depth(self, compression_type):
        if compression_type == "zlib_ushort":
            return self.decompress_depth_zlib()
        else:
            raise

    def decompress_depth_zlib(self):
        return zlib.decompress(self.depth_data)

    def decompress_color(self, compression_type):
        if compression_type == "jpeg":
            return self.decompress_color_jpeg()
        else:
            raise

    def decompress_color_jpeg(self):
        return imageio.imread(self.color_data)


class IMUFrame:
    def load(self, file_handle):
        self.rotation_rate = np.asarray(
            struct.unpack("d" * 3, file_handle.read(3 * 8)), dtype=np.float64
        )
        self.acceleration = np.asarray(
            struct.unpack("d" * 3, file_handle.read(3 * 8)), dtype=np.float64
        )
        self.magnetic_field = np.asarray(
            struct.unpack("d" * 3, file_handle.read(3 * 8)), dtype=np.float64
        )
        self.attitude = np.asarray(
            struct.unpack("d" * 3, file_handle.read(3 * 8)), dtype=np.float64
        )
        self.gravity = np.asarray(
            struct.unpack("d" * 3, file_handle.read(3 * 8)), dtype=np.float64
        )
        self.timestamp = struct.unpack("Q", file_handle.read(8))[0]


class SensorData:
    def __init__(self, filename):
        self.version = 4
        self.load(filename)

    def load(self, filename):
        with open(filename, "rb") as f:
            version = struct.unpack("I", f.read(4))[0]
            assert self.version == version
            strlen = struct.unpack("Q", f.read(8))[0]
            self.sensor_name = b"".join(struct.unpack("c" * strlen, f.read(strlen)))
            self.intrinsic_color = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.extrinsic_color = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.intrinsic_depth = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.extrinsic_depth = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.color_compression_type = COMPRESSION_TYPE_COLOR[
                struct.unpack("i", f.read(4))[0]
            ]
            self.depth_compression_type = COMPRESSION_TYPE_DEPTH[
                struct.unpack("i", f.read(4))[0]
            ]
            self.color_width = struct.unpack("I", f.read(4))[0]
            self.color_height = struct.unpack("I", f.read(4))[0]
            self.depth_width = struct.unpack("I", f.read(4))[0]
            self.depth_height = struct.unpack("I", f.read(4))[0]
            self.depth_shift = struct.unpack("f", f.read(4))[0]
            num_frames = struct.unpack("Q", f.read(8))[0]
            self.frames = []
            for i in range(num_frames):
                frame = RGBDFrame()
                frame.load(f)
                self.frames.append(frame)

            num_imu_frames = struct.unpack("Q", f.read(8))[0]
            self.imu_frames = []
            for i in range(num_imu_frames):
                imu_frame = IMUFrame()
                imu_frame.load(f)
                self.imu_frames.append(imu_frame)

    def export_depth_images(self, output_path, image_size=None, frame_skip=1):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print(
            "exporting", len(self.frames) // frame_skip, " depth frames to", output_path
        )
        for f in range(0, len(self.frames), frame_skip):
            depth_data = self.frames[f].decompress_depth(self.depth_compression_type)
            depth = np.fromstring(depth_data, dtype=np.uint16).reshape(
                self.depth_height, self.depth_width
            )
            if image_size is not None:
                depth = cv2.resize(
                    depth,
                    (image_size[0], image_size[1]),
                    interpolation=cv2.INTER_NEAREST,
                )
            formatted_frame_no = format_frame_number_with_leading_zeros(f)
            with open(
                os.path.join(output_path, formatted_frame_no + ".png"), "wb"
            ) as fh:  # write 16-bit
                writer = png.Writer(
                    width=depth.shape[1], height=depth.shape[0], bitdepth=16
                )
                depth = depth.reshape(-1, depth.shape[1]).tolist()
                writer.write(fh, depth)

    def export_color_images(self, output_path, image_size=None, frame_skip=1):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print(
            "exporting", len(self.frames) // frame_skip, "color frames to", output_path
        )
        for f in range(0, len(self.frames), frame_skip):
            color = self.frames[f].decompress_color(self.color_compression_type)
            if image_size is not None:
                color = cv2.resize(
                    color,
                    (image_size[0], image_size[1]),
                    interpolation=cv2.INTER_AREA,
                )
            formatted_frame_no = format_frame_number_with_leading_zeros(f)
            imageio.imwrite(
                os.path.join(output_path, formatted_frame_no + ".jpg"), color
            )

    def save_mat_to_file(self, matrix, filename):
        with open(filename, "w") as f:
            for line in matrix:
                np.savetxt(f, line[np.newaxis], fmt="%f")

    def export_poses(self, output_path, frame_skip=1):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print(
            "exporting", len(self.frames) // frame_skip, "camera poses to", output_path
        )
        for f in range(0, len(self.frames), frame_skip):
            formatted_frame_no = format_frame_number_with_leading_zeros(f)
            self.save_mat_to_file(
                self.frames[f].camera_to_world,
                os.path.join(output_path, formatted_frame_no + ".txt"),
            )

    def export_intrinsics(self, output_path, image_size=None):
        """Export camera intrinsics as .txt file

        If image_size is passed, adjust the camera intrinsics to match the new image size
        """
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print("exporting camera intrinsics to", output_path)
        if image_size is not None:
            # Adjust intrinsics matrix
            adjusted_intrinsic_color = adjust_intrinsic_matrix(
                self.intrinsic_color,
                (self.color_width, self.color_height),
                (image_size[0], image_size[1]),
            )
            self.save_mat_to_file(
                adjusted_intrinsic_color,
                os.path.join(
                    output_path,
                    "intrinsic_color.txt",
                ),
            )

            adjusted_intrinsic_depth = adjust_intrinsic_matrix(
                self.intrinsic_depth,
                (self.depth_width, self.depth_height),
                (image_size[0], image_size[1]),
            )

            self.save_mat_to_file(
                adjusted_intrinsic_depth,
                os.path.join(output_path, "intrinsic_depth.txt"),
            )

        else:
            self.save_mat_to_file(
                self.intrinsic_color, os.path.join(output_path, "intrinsic_color.txt")
            )
            self.save_mat_to_file(
                self.intrinsic_depth, os.path.join(output_path, "intrinsic_depth.txt")
            )
        self.save_mat_to_file(
            self.extrinsic_color, os.path.join(output_path, "extrinsic_color.txt")
        )
        self.save_mat_to_file(
            self.extrinsic_depth, os.path.join(output_path, "extrinsic_depth.txt")
        )

    def export_imu_timestamps(self, output_file_path):
        field_names = ["FrameID", "TimeStamp"]
        with open(output_file_path, "w") as f:
            writer = csv.DictWriter(f, fieldnames=field_names)
            writer.writeheader()
            for i in range(len(self.imu_frames)):
                writer.writerow(
                    {"FrameID": i, "TimeStamp": self.imu_frames[i].timestamp}
                )



def main():
    # If not specified use the same directory as the .sens file
    if opt.output_path is None:
        opt.output_path = os.path.dirname(opt.filename)
    os.makedirs(opt.output_path, exist_ok=True)
    # load the data
    sys.stdout.write("loading %s..." % opt.filename)
    sd = SensorData(opt.filename)
    sys.stdout.write("loaded!\n")
    if opt.export_color:
        sd.export_color_images(
            os.path.join(opt.output_path, "color"),
            image_size=opt.image_size,
        )
    if opt.export_depth:
        sd.export_depth_images(os.path.join(opt.output_path, "depth"), opt.image_size)
    if opt.export_pose:
        sd.export_poses(os.path.join(opt.output_path, "pose"))
    if opt.export_intrinsic:
        sd.export_intrinsics(os.path.join(opt.output_path, "intrinsic"), opt.image_size)
    if opt.export_timestamps:
        sd.export_imu_timestamps(os.path.join(opt.output_path, "timestamps.csv"))


SCANNETV2_TO_NYU40 = {
    0: 0,
    1: 1,
    2: 5,
    22: 23,
    3: 2,
    5: 8,
    1163: 40,
    16: 9,
    4: 7,
    56: 39,
    13: 18,
    15: 11,
    41: 22,
    26: 29,
    161: 8,
    19: 40,
    7: 3,
    9: 14,
    8: 15,
    10: 5,
    31: 27,
    6: 6,
    14: 34,
    48: 40,
    28: 35,
    11: 4,
    18: 10,
    71: 19,
    21: 16,
    40: 40,
    52: 30,
    96: 39,
    29: 3,
    49: 40,
    23: 5,
    63: 40,
    24: 7,
    17: 33,
    47: 37,
    32: 21,
    46: 40,
    65: 40,
    97: 39,
    34: 32,
    38: 40,
    33: 25,
    75: 3,
    36: 17,
    64: 40,
    101: 40,
    130: 40,
    27: 24,
    44: 7,
    131: 40,
    55: 28,
    42: 36,
    59: 40,
    159: 12,
    74: 5,
    82: 40,
    1164: 3,
    93: 40,
    77: 40,
    67: 39,
    128: 1,
    50: 40,
    35: 12,
    69: 38,
    100: 40,
    62: 38,
    105: 38,
    1165: 1,
    165: 24,
    76: 40,
    230: 40,
    54: 40,
    125: 38,
    72: 40,
    68: 39,
    145: 38,
    157: 40,
    1166: 40,
    132: 40,
    1167: 8,
    232: 38,
    134: 40,
    51: 39,
    250: 40,
    1168: 38,
    342: 38,
    89: 38,
    103: 40,
    99: 39,
    95: 38,
    154: 38,
    140: 20,
    1169: 39,
    193: 38,
    116: 39,
    202: 40,
    73: 40,
    78: 38,
    1170: 40,
    79: 26,
    80: 31,
    141: 38,
    57: 39,
    102: 40,
    261: 40,
    118: 40,
    136: 38,
    98: 40,
    1171: 38,
    170: 40,
    1172: 40,
    1173: 3,
    221: 40,
    570: 37,
    138: 40,
    168: 40,
    276: 8,
    106: 40,
    214: 40,
    323: 40,
    58: 38,
    86: 13,
    399: 40,
    121: 40,
    185: 40,
    300: 40,
    180: 40,
    163: 40,
    66: 40,
    208: 40,
    112: 40,
    540: 29,
    395: 38,
    166: 40,
    122: 39,
    120: 38,
    107: 38,
    283: 40,
    88: 40,
    90: 39,
    177: 39,
    1174: 40,
    562: 40,
    1175: 40,
    1156: 12,
    84: 38,
    104: 39,
    229: 40,
    70: 39,
    325: 40,
    169: 40,
    331: 40,
    87: 39,
    488: 40,
    776: 40,
    370: 40,
    191: 38,
    748: 40,
    242: 40,
    45: 7,
    417: 2,
    188: 38,
    1176: 40,
    1177: 39,
    1178: 38,
    110: 39,
    148: 40,
    155: 39,
    572: 40,
    1179: 38,
    392: 40,
    1180: 39,
    609: 38,
    1181: 40,
    195: 40,
    581: 39,
    1182: 40,
    1183: 40,
    139: 40,
    1184: 5,
    1185: 40,
    156: 38,
    408: 40,
    213: 39,
    1186: 40,
    1187: 40,
    1188: 11,
    115: 40,
    1189: 40,
    304: 40,
    1190: 40,
    312: 40,
    233: 39,
    286: 40,
    264: 40,
    1191: 4,
    356: 40,
    25: 39,
    750: 40,
    269: 40,
    307: 39,
    410: 39,
    730: 38,
    216: 40,
    1192: 38,
    119: 40,
    682: 40,
    434: 40,
    126: 39,
    919: 40,
    85: 39,
    1193: 7,
    108: 7,
    135: 40,
    1194: 40,
    432: 40,
    53: 40,
    1195: 40,
    111: 40,
    305: 38,
    1125: 40,
    1196: 40,
    1197: 21,
    1198: 40,
    1199: 40,
    1200: 40,
    378: 40,
    591: 40,
    92: 40,
    1098: 40,
    291: 40,
    1063: 38,
    1135: 40,
    189: 40,
    245: 40,
    194: 40,
    1201: 38,
    386: 40,
    1202: 39,
    857: 40,
    452: 40,
    1203: 40,
    346: 40,
    152: 38,
    83: 40,
    1204: 1,
    726: 40,
    61: 40,
    39: 18,
    1117: 39,
    1205: 40,
    415: 40,
    1206: 40,
    153: 39,
    1207: 40,
    129: 39,
    220: 40,
    1208: 8,
    231: 40,
    1209: 39,
    1210: 40,
    117: 38,
    822: 39,
    238: 40,
    143: 39,
    1211: 40,
    228: 40,
    494: 4,
    226: 40,
    91: 39,
    1072: 37,
    435: 40,
    345: 40,
    893: 40,
    621: 40,
    1212: 40,
    297: 40,
    1213: 23,
    1214: 40,
    1215: 38,
    529: 40,
    1216: 38,
    1217: 40,
    1218: 11,
    1219: 38,
    1220: 38,
    525: 39,
    204: 40,
    693: 40,
    179: 35,
    1221: 40,
    1222: 40,
    1223: 40,
    1224: 40,
    1225: 22,
    1226: 40,
    1227: 39,
    571: 40,
    1228: 40,
    556: 40,
    280: 40,
    1229: 40,
    1230: 37,
    1231: 40,
    1232: 37,
    746: 40,
    1233: 40,
    1234: 40,
    144: 40,
    282: 39,
    167: 40,
    1235: 40,
    1236: 40,
    1237: 40,
    234: 39,
    563: 40,
    1238: 37,
    1239: 40,
    1240: 40,
    366: 40,
    816: 40,
    1241: 40,
    719: 40,
    284: 40,
    1242: 39,
    247: 40,
    1243: 1,
    1244: 39,
    1245: 29,
    1246: 40,
    1247: 40,
    592: 40,
    385: 3,
    1248: 40,
    1249: 40,
    133: 40,
    301: 38,
    1250: 40,
    379: 38,
    1251: 40,
    450: 40,
    1252: 37,
    316: 40,
    1253: 29,
    1254: 31,
    461: 40,
    1255: 40,
    1256: 39,
    599: 40,
    281: 40,
    1257: 33,
    1258: 40,
    1259: 40,
    319: 40,
    1260: 40,
    1261: 40,
    546: 40,
    1262: 40,
    1263: 40,
    1264: 37,
    1265: 40,
    1266: 40,
    1267: 20,
    1268: 40,
    1269: 40,
    689: 40,
    1270: 39,
    1271: 29,
    1272: 40,
    354: 39,
    339: 40,
    1009: 40,
    1273: 40,
    1274: 40,
    1275: 40,
    361: 40,
    1276: 40,
    326: 39,
    1277: 40,
    1278: 40,
    1279: 40,
    212: 40,
    1280: 40,
    1281: 40,
    794: 40,
    1282: 40,
    955: 40,
    387: 40,
    523: 40,
    389: 39,
    1283: 15,
    146: 38,
    372: 40,
    289: 39,
    440: 37,
    321: 40,
    976: 38,
    1284: 40,
    1285: 40,
    357: 27,
    1286: 40,
    1287: 40,
    365: 40,
    1288: 37,
    81: 39,
    1289: 40,
    1290: 39,
    948: 40,
    174: 40,
    1028: 40,
    1291: 5,
    1292: 40,
    1005: 40,
    235: 38,
    1293: 40,
    1294: 40,
    1295: 38,
    1296: 40,
    1297: 37,
    1298: 40,
    1299: 29,
    1300: 40,
    1301: 21,
    1051: 40,
    566: 39,
    1302: 40,
    1062: 24,
    1303: 21,
    1304: 40,
    1305: 40,
    1306: 40,
    298: 40,
    1307: 40,
    1308: 40,
    1309: 40,
    43: 39,
    1310: 38,
    593: 40,
    1311: 40,
    1312: 40,
    749: 35,
    623: 40,
    1313: 6,
    265: 40,
    1314: 40,
    1315: 40,
    448: 38,
    257: 40,
    1316: 15,
    786: 4,
    801: 40,
    972: 40,
    1317: 40,
    1318: 40,
    657: 29,
    561: 40,
    513: 38,
    411: 39,
    1122: 38,
    922: 40,
    518: 40,
    814: 40,
    1319: 40,
    1320: 40,
    649: 8,
    607: 40,
    819: 40,
    1321: 40,
    1322: 3,
    227: 40,
    817: 40,
    712: 40,
    1323: 40,
    1324: 40,
    673: 29,
    459: 40,
    643: 40,
    765: 39,
    1008: 40,
    225: 40,
    1083: 40,
    813: 40,
    1145: 35,
    796: 40,
    1325: 40,
    363: 39,
    1326: 40,
    997: 40,
    1327: 40,
    1328: 40,
    1329: 40,
    182: 40,
    1330: 40,
    1331: 40,
    1332: 40,
    1333: 40,
    939: 40,
    1334: 40,
    480: 37,
    907: 40,
    1335: 15,
    1336: 40,
    829: 40,
    947: 1,
    1116: 40,
    733: 40,
    123: 40,
    506: 37,
    569: 8,
    1337: 40,
    1338: 5,
    1339: 40,
    1340: 38,
    851: 39,
    142: 40,
    436: 40,
    1341: 39,
    1342: 21,
    885: 5,
    815: 3,
    401: 40,
    1343: 40,
    1344: 40,
    1345: 8,
    160: 38,
    1126: 40,
    1346: 40,
    332: 40,
    397: 40,
    551: 40,
    1347: 2,
    1348: 40,
    803: 40,
    484: 39,
    1349: 4,
    1350: 40,
    222: 7,
    1351: 39,
    1352: 40,
    828: 40,
    1353: 40,
    612: 40,
    1354: 40,
    1355: 7,
    1356: 37,
    1357: 40,
}

if __name__ == "__main__":
    main()