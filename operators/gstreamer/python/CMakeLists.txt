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

include(pybind11_add_holohub_module)
pybind11_add_holohub_module(
    CPP_CMAKE_TARGET holoscan_gstreamer_bridge
    CLASS_NAME "GstVideoRecorderOp"
    SOURCES gst_video_recorder_op.cpp
)

if(BUILD_TESTING)
  find_package(holoscan 1.0 REQUIRED CONFIG PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

  add_test(
    NAME holoscan_gstreamer_bridge_python_import_test
    COMMAND python3 -c
            "from holohub.holoscan_gstreamer_bridge import GstVideoRecorderOp; assert GstVideoRecorderOp.__name__ == 'GstVideoRecorderOp'; assert GstVideoRecorderOp.__doc__"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  set_property(
    TEST holoscan_gstreamer_bridge_python_import_test
    PROPERTY ENVIRONMENT
      "PYTHONPATH=${GXF_LIB_DIR}/../python/lib:${CMAKE_BINARY_DIR}/python/lib"
  )
endif()
