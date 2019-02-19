# TVM/AOCL Research Division

This section of distributed-deep-learning is a compository of all research and testing pertaining to the AOCL backend of TVM. Current version supports generating AOCL code for vector addition to be run on an Intel a10gx FPGA.

## Getting Started

1) Navigate to your FPGA directory and run setup scripts
> for edge-1 machine see /home/tools/altera/17.1-pro/hld/init_opencl.sh

### Prerequisites

1) Build and install TVM with AOCL, OpenCL, and LLVM enabled
> see https://docs.tvm.ai/install/from_source.html
2) Install FPGA device driver
3) Install BSP for FPGA

### Installing

## Running the tests

To run the vector addition simply navigate to aocl directory and enter
``` source run.sh ```

To switch between the FPGA emulator and actual FPGA device you can change the ```target``` within both run.py and build.py
- for tests using the emulator set ```tgt="aocl_sw_emu"```
- for tests using physical FPGA set ```tgt="aocl -device=a10gx"``` or respective device name 

Then modify run.sh
- for tests using the emulator set ```"export CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1"```
- for tests using physical FPGA set ```unset CL_CONTEXT_EMULATOR_DEVICE_FGPA``` (only necessary if previously ran run.sh with emulator settings) 

### Break down into end to end tests
To test whether the TVM AOCL backend is even functioning, run run.sh with build.py, run.py, and run.sh configued to target the emulator. If TVM is built and configured correctly, AOCL code for vector addition will be output, verifying all functionality.


## Deployment

For difficulties deploying on physical FPGA verify that
- An ICD file exists so that the OpenCL platform can be found.
> for edge-1 see /home/tools/altera/17.1-pro/hld/linux64/lib/libalteracl.so
- An FCD file exists so that your FPGA device can be found.
> for edge-1 see /home/tools/altera/17.1-pro/hld/board/a10_ref/linux64/lib/libaltera_a10_ref_mmd.so

## Built With
TVM 0.5.dev

## Contributing


## Versioning

(Version 1.0) AOCL vector addition

## Authors
Jared Nathenson (https://github.com/Jnathenson96)

## License

## Acknowledgments

* Ktabata (https://github.com/ktabata) for providing vector addition source code and providing AOCL backend for TVM
* Saman Biookaghazadeh (https://github.com/saman-aghazadeh) for inspiration and project sponsorship.