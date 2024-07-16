#!/usr/bin/env cs_python

import argparse
import json
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder # pylint: disable=no-name-in-module

# Read arguments
parser = argparse.ArgumentParser()
parser.add_argument('--name', help="the test compile output dir")
parser.add_argument('--cmaddr', help="IP:port for CS system")
args = parser.parse_args()

# Get array dimension from compile metadata
with open(f"{args.name}/out.json", encoding='utf-8') as json_file:
  compile_data = json.load(json_file)

# Array dimension
M = int(compile_data['params']['M'])

# Construct input z
# Elements [:M] goes to left PE; elements [M:] goes to right PE
north_z = np.ones(shape=M*2, dtype=np.float32)
north_z[0] = 0.0;
north_z[M] = 2.0;

# Will be used to store result
north_z_result = np.zeros([2*M], dtype=np.float32)

print("Before memcpy_d2h, north z is: ")
print("PE(0,0): ", north_z[:M], "    PE(1,0): ", north_z[M:])

print("\nBegin run...")

# Construct a runner using SdkRuntime
runner = SdkRuntime(args.name, cmaddr=args.cmaddr)

# Get symbols for z on device
z_symbol = runner.get_id('z')

# Load and run the program
runner.load()
runner.run()

# Launch all memcpy and launch commands asynchronously
# so that they are queued by memcpy

# Copy z into north worker PEs
runner.memcpy_h2d(z_symbol, north_z, 0, 0, 2, 1, M, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=True)

# Launch ctrl fxn on ctrl PE, which sends control wavelets along cmd fan
runner.launch('launch_ctrl', nonblock=True)

# Copy z back from north worker PEs
runner.memcpy_d2h(north_z_result, z_symbol, 0, 0, 2, 1, M, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=True)

# Wait until tasks are done and stop the program
runner.stop()
print("Stopped.\n")

print("After memcpyh2d, north z is: ")
print("PE(0,0): ", north_z_result[:M], "    PE(1,0): ", north_z_result[M:])
