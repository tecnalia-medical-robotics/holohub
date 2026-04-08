# AJA Video Capture

Minimal example to demonstrate the use of the aja source operator to capture device input and stream to holoviz operator.

*Visit the [SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/aja_setup.html) to setup the AJA Card.*

## Quick Start

```bash
./holohub run --build-args="--target holohub-aja" aja_video_capture --language <cpp/python>
```

## Overlay demo

The default Python entrypoint runs the basic AJA capture application:

```bash
./holohub run --build-args="--target holohub-aja" aja_video_capture --language=python
```

To launch the overlay demonstration instead, run the `overlay` mode:

```bash
./holohub run --build-args="--target holohub-aja" aja_video_capture overlay --language=python
```

This mode launches `aja_capture_overlay.py`.
It requires an overlay-capable AJA setup and uses the overlay configuration provided by `aja_capture_overlay.yaml`.

## Settings

 To evaluate the AJA example using alternative resolutions, you may modify the aja_capture.yaml configuration file as needed. For instance, to test a resolution format of 1280 x 720 at 60 Hz, you can specify the following parameters in the aja section of the configuration :

```yaml
aja:
  width: 1280
  height: 720
  framerate: 60
```

## Migration Notes

Holoscan SDK AJA support is migrated from the core Holoscan SDK library to the HoloHub community repository in Holoscan SDK v3.0.0.
Projects depending on AJA support should accordingly update include and linking paths to reference HoloHub.

C++/CMake projects should update `holoscan::ops::aja` to `holoscan::aja`

Python projects should update `import holoscan.operators.AJASourceOp` to `import holohub.aja_source.AJASourceOp`
