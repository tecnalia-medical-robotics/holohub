"""
SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""  # noqa: E501

import os

import cupy as cp
import holoscan as hs
from holoscan.core import Application, OperatorSpec
from holoscan.gxf import Entity
from holoscan.operators import HolovizOp

from holohub.aja_source import AJASourceOp


def _make_circle_rgba(
    width: int,
    height: int,
    radius: int,
    center_x: int,
    center_y: int,
    color: tuple[int, int, int, int],
) -> cp.ndarray:
    y = cp.arange(height).reshape(-1, 1)
    x = cp.arange(width).reshape(1, -1)
    dist2 = (x - center_x) ** 2 + (y - center_y) ** 2
    mask = dist2 <= radius**2
    img = cp.zeros((height, width, 4), dtype=cp.uint8)
    img[mask] = cp.array(color, dtype=cp.uint8)
    return img

class CircleImageOp(hs.core.Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")
        spec.param("out_tensor_name", "circle")
        spec.param("width", 1920)
        spec.param("height", 1080)
        spec.param("radius", 100)
        spec.param("center_x", 960)
        spec.param("center_y", 540)
        spec.param("color", (0, 255, 0, 255))


    def _validate_circle_config(self):
        if self.width <= 0:
            raise ValueError(f"width must be > 0, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"height must be > 0, got {self.height}")
        if self.radius < 0:
            raise ValueError(f"radius must be >= 0, got {self.radius}")
        if not 0 <= self.center_x < self.width:
            raise ValueError(
                f"center_x must be in [0, {self.width}), got {self.center_x}"
            )
        if not 0 <= self.center_y < self.height:
            raise ValueError(
                f"center_y must be in [0, {self.height}), got {self.center_y}"
            )

    def compute(self, op_input, op_output, context):
        op_input.receive("in")

        self._validate_circle_config()

        # Intentionally regenerate the overlay every frame.
        # In real-time pipelines this operator may represent dynamic
        # analysis output rather than a static precomputed asset.
        img_gpu = _make_circle_rgba(
            width=self.width,
            height=self.height,
            radius=self.radius,
            center_x=self.center_x,
            center_y=self.center_y,
            color=tuple(self.color),
        )
        msg = Entity(context)
        msg.add(hs.as_tensor(img_gpu), self.out_tensor_name)
        op_output.emit(msg, "out")

class AJACaptureApp(Application):
    """
    Example of an application that uses the following operators:

    - AJASourceOp
    - HolovizOp

    The AJASourceOp reads frames from an AJA input device and sends it to the HolovizOp.
    The HolovizOp displays the frames.
    """

    def compose(self):

        source = AJASourceOp(self, name="aja", **self.kwargs("aja"))

        visualizer = HolovizOp(self, name="holoviz", **self.kwargs("holoviz"))
        circle_generator = CircleImageOp(
            self,
            name="circle_generator",
            **self.kwargs("circle_generator"),
        )
        holoviz_to_videobuffer = HolovizOp(self, name="holoviz_to_videobuffer", **self.kwargs("holoviz_to_videobuffer"))
        holoviz_aja_consumer = HolovizOp(self, name="holoviz_aja_consumer", **self.kwargs("holoviz_aja_consumer"))

        self.add_flow(source, visualizer, {("video_buffer_output", "receivers")})
        self.add_flow(source, circle_generator, {("video_buffer_output", "in")})
        self.add_flow(circle_generator, holoviz_to_videobuffer, {("out", "receivers")})
        self.add_flow(holoviz_to_videobuffer, source, {("render_buffer_output", "overlay_buffer_input")})
        self.add_flow(source, holoviz_aja_consumer, {("overlay_buffer_output", "receivers")})


def main(config_file):
    app = AJACaptureApp()
    # if the --config command line argument was provided, it will override this config_file
    app.config(config_file)
    app.run()


if __name__ == "__main__":
    config_file = os.path.join(os.path.dirname(__file__), "aja_capture_external_overlay.yaml")
    main(config_file=config_file)
