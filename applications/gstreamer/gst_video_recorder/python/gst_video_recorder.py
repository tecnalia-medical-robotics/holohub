# SPDX-FileCopyrightText: Copyright (c) 2025-2026, TECNALIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, Optional

import numpy as np

from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import FormatConverterOp, V4L2VideoCaptureOp
from holoscan.resources import BlockMemoryPool, MemoryStorageType, UnboundedAllocator

from holohub.holoscan_gstreamer_bridge import GstVideoRecorderOp


def parse_pattern(value: str) -> str:
    mapping = {
        "0": "gradient",
        "1": "checkerboard",
        "2": "colorbars",
        "gradient": "gradient",
        "checkerboard": "checkerboard",
        "colorbars": "colorbars",
        "bars": "colorbars",
    }
    key = str(value).strip().lower()
    if key not in mapping:
        raise argparse.ArgumentTypeError(
            "invalid pattern; use 0|1|2 or gradient|checkerboard|colorbars"
        )
    return mapping[key]


def parse_v4l2_pixel_format(value: str) -> str:
    mapping = {
        "auto": "auto",
        "ab24": "AB24",
        "rgb3": "RGB3",
        "mjpg": "MJPG",
        "mjpeg": "MJPG",
        "yuyv": "YUYV",
    }
    key = str(value).strip().lower()
    if key not in mapping:
        raise argparse.ArgumentTypeError(
            "invalid --pixel-format; use auto, AB24, RGB3, MJPG/MJPEG, or YUYV"
        )
    return mapping[key]


def parse_key_value_properties(items: list[str]) -> Dict[str, Any]:
    def convert_scalar(raw: str) -> Any:
        text = raw.strip()
        lowered = text.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"

        try:
            return int(text)
        except ValueError:
            pass

        try:
            return float(text)
        except ValueError:
            pass

        return text

    props: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"invalid --property '{item}', expected KEY=VALUE"
            )
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise argparse.ArgumentTypeError("property key cannot be empty")
        props[key] = convert_scalar(value)
    return props


class PatternGeneratorOp(Operator):
    """Emit RGBA frames for pattern-generator mode."""

    def __init__(
        self,
        fragment,
        *args,
        width: int = 1920,
        height: int = 1080,
        pattern: str = "gradient",
        **kwargs,
    ):
        self.width = int(width)
        self.height = int(height)
        self.pattern = pattern
        self.frame_index = 0
        self._x: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("output")

    def start(self):
        self._y, self._x = np.indices((self.height, self.width), dtype=np.uint16)

    def _gradient(self) -> np.ndarray:
        assert self._x is not None and self._y is not None
        frame = np.empty((self.height, self.width, 4), dtype=np.uint8)
        frame[..., 0] = ((self._x + self.frame_index * 2) % 256).astype(np.uint8)
        frame[..., 1] = ((self._y + self.frame_index * 3) % 256).astype(np.uint8)
        frame[..., 2] = (
            ((self._x // 2) + (self._y // 2) + self.frame_index * 5) % 256
        ).astype(np.uint8)
        frame[..., 3] = 255
        return frame

    def _checkerboard(self) -> np.ndarray:
        assert self._x is not None and self._y is not None
        tile = max(16, min(self.width, self.height) // 12)
        shifted_x = (self._x + self.frame_index * 8) // tile
        shifted_y = (self._y + self.frame_index * 4) // tile
        board = ((shifted_x + shifted_y) % 2).astype(np.uint8) * 255

        frame = np.empty((self.height, self.width, 4), dtype=np.uint8)
        frame[..., 0] = board
        frame[..., 1] = np.roll(board, shift=tile // 2, axis=1)
        frame[..., 2] = 255 - board
        frame[..., 3] = 255
        return frame

    def _colorbars(self) -> np.ndarray:
        bar_colors = np.array(
            [
                [255, 255, 255, 255],  # white
                [255, 255, 0, 255],    # yellow
                [0, 255, 255, 255],    # cyan
                [0, 255, 0, 255],      # green
                [255, 0, 255, 255],    # magenta
                [255, 0, 0, 255],      # red
                [0, 0, 255, 255],      # blue
            ],
            dtype=np.uint8,
        )

        frame = np.empty((self.height, self.width, 4), dtype=np.uint8)
        bar_width = max(1, self.width // len(bar_colors))
        for index, color in enumerate(bar_colors):
            start = index * bar_width
            end = self.width if index == len(bar_colors) - 1 else (index + 1) * bar_width
            frame[:, start:end, :] = color

        shift = (self.frame_index * 4) % max(1, self.width)
        if shift:
            frame = np.roll(frame, shift=shift, axis=1)
        return frame

    def compute(self, op_input, op_output, context):
        if self.pattern == "gradient":
            frame = self._gradient()
        elif self.pattern == "checkerboard":
            frame = self._checkerboard()
        elif self.pattern == "colorbars":
            frame = self._colorbars()
        else:
            raise RuntimeError(f"unsupported pattern '{self.pattern}'")

        self.frame_index += 1
        op_output.emit({"frame": frame}, "output")


class GstVideoRecorderApp(Application):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args

    def _source_condition_args(self) -> list[Any]:
        if self.args.count > 0:
            return [CountCondition(self, self.args.count)]
        return []

    def compose(self):
        condition_args = self._source_condition_args()

        recorder = GstVideoRecorderOp(
            self,
            encoder=self.args.encoder,
            format=self.args.format,
            framerate=self.args.framerate,
            max_buffers=self.args.max_buffers,
            block=self.args.block,
            filename=self.args.output,
            properties=self.args.properties,
            name="gst_video_recorder",
        )

        if self.args.source == "pattern":
            source = PatternGeneratorOp(
                self,
                *condition_args,
                width=self.args.width,
                height=self.args.height,
                pattern=self.args.pattern,
                name="pattern_source",
            )
            self.add_flow(source, recorder, {("output", "input")})
        elif self.args.source == "v4l2":
            pool = UnboundedAllocator(self, name="pool")

            source = V4L2VideoCaptureOp(
                self,
                *condition_args,
                allocator=pool,
                device=self.args.device,
                width=self.args.width,
                height=self.args.height,
                frame_rate=float(self.args.fps),
                pixel_format=self.args.pixel_format,
                name="v4l2_source",
            )
            format_converter = FormatConverterOp(
                self,
                name="format_converter",
                in_dtype="rgba8888",
                out_dtype="rgba8888",
                pool=pool,
            )

            self.add_flow(source, format_converter, {("signal", "source_video")})
            self.add_flow(format_converter, recorder, {("tensor", "input")})

        else:
            raise RuntimeError(f"unsupported source '{self.args.source}'")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Holohub Python sample for GstVideoRecorderOp."
    )

    # General options
    parser.add_argument(
        "--source",
        choices=("pattern", "v4l2"),
        default="pattern",
        help="input source type",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output.mp4",
        help="output video filename",
    )
    parser.add_argument(
        "--encoder",
        default="x264",
        help="video encoder (for example: x264, x265, nvh264, nvh265)",
    )
    parser.add_argument(
        "--format",
        default="RGBA",
        help="input frame format passed to GstVideoRecorderOp",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="nominal frames per second",
    )
    parser.add_argument(
        "--framerate",
        default=None,
        help='GStreamer framerate string (for example "30/1", "30000/1001", "0/1")',
    )
    parser.add_argument(
        "--count",
        "--frames",
        dest="count",
        type=int,
        default=300,
        help="number of frames to record; use 0 or negative for unlimited",
    )
    parser.add_argument(
        "--max-buffers",
        type=int,
        default=10,
        help="maximum number of queued buffers for GstVideoRecorderOp",
    )

    block_group = parser.add_mutually_exclusive_group()
    block_group.add_argument(
        "--block",
        dest="block",
        action="store_true",
        default=True,
        help="block when the recorder queue is full",
    )
    block_group.add_argument(
        "--no-block",
        dest="block",
        action="store_false",
        help="do not block when the recorder queue is full",
    )

    parser.add_argument(
        "--property",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra GStreamer encoder property; may be repeated",
    )

    # Resolution options
    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="frame width in pixels",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="frame height in pixels",
    )

    # V4L2 options via built-in Holoscan operator
    parser.add_argument(
        "--device",
        default="/dev/video0",
        help="V4L2 device path",
    )
    parser.add_argument(
        "--pixel-format",
        type=parse_v4l2_pixel_format,
        default="auto",
        help="V4L2 fourcc preference: auto, AB24, RGB3, MJPG/MJPEG, or YUYV",
    )

    # Pattern generator options
    parser.add_argument(
        "--pattern",
        type=parse_pattern,
        default="gradient",
        help="pattern type: 0/gradient, 1/checkerboard, 2/colorbars",
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.width <= 0 or args.height <= 0:
        raise SystemExit("--width and --height must be positive integers")
    if args.max_buffers < 0:
        raise SystemExit("--max-buffers must be >= 0")
    if args.fps <= 0:
        raise SystemExit("--fps must be > 0")
    if args.framerate is None:
        args.framerate = f"{args.fps}/1"
    args.properties = parse_key_value_properties(args.property)


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_args(args)

    try:
        app = GstVideoRecorderApp(args)
        app.run()
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
