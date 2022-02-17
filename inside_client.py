#!/usr/bin/env python
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import numpy as np
import os
import sys
import cv2
from builtins import range
from ctypes import *

import PIL.Image as pil_image

import tritonclient.grpc as grpcclient
from tritonclient import utils
import tritonclient.utils.shared_memory as shm


# numpy to numpy
def call_triton_inference(image:np.ndarray, model_name:str, scale:int, channel:int) -> np.ndarray:
    url = "localhost:8001"
    verbose = False
    model_version = "1"
    try:
        triton_client = grpcclient.InferenceServerClient(url=url,
                                                         verbose=verbose)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    # To make sure no shared memory regions are registered with the
    # server.
    triton_client.unregister_system_shared_memory()
    triton_client.unregister_cuda_shared_memory()

    # We use a simple model that takes 2 input tensors of 16 integers
    # each and returns 2 output tensors of 16 integers each. One
    # output tensor is the element-wise sum of the inputs and one
    # output is the element-wise difference.

    # Create the data for the two input tensors. Initialize the first
    # to unique integers and the second to all ones.
    input0_data = image

    input_byte_size = input0_data.size * input0_data.itemsize
    output_byte_size = input_byte_size * (scale ** 2 + 1)

    # Create Output and Output1 in Shared Memory and store shared memory handles
    shm_op0_handle = shm.create_shared_memory_region("output_data",
                                                     "/output_simple",
                                                     output_byte_size)
    # Register Output and Output1 shared memory with Triton Server
    triton_client.register_system_shared_memory("output_data",
                                                "/output_simple",
                                                output_byte_size)

    # Create Input0 and Input1 in Shared Memory and store shared memory handles
    shm_ip0_handle = shm.create_shared_memory_region("input0_data",
                                                     "/input0_simple",
                                                     input_byte_size)
    # Put input data values into shared memory
    shm.set_shared_memory_region(shm_ip0_handle, [input0_data])

    # Register Input0 and Input1 shared memory with Triton Server
    triton_client.register_system_shared_memory("input0_data", "/input0_simple",
                                                input_byte_size)
    # Set the parameters to use data from shared memory
    inputs = []
    inputs.append(grpcclient.InferInput('input0', image.shape, "FP32"))
    inputs.append(grpcclient.InferInput('input1', [1], "INT32"))
    inputs[0].set_shared_memory("input0_data", input_byte_size)
    inputs[1].set_data_from_numpy(np.array([channel]).astype(np.int32))

    outputs = []
    outputs.append(grpcclient.InferRequestedOutput('output'))
    outputs[-1].set_shared_memory("output_data", output_byte_size)

    results = triton_client.infer(model_name=model_name,
                                  inputs=inputs,
                                  outputs=outputs)

    # Read results from the shared memory.
    output = results.get_output("output")
    if output is not None:
        output_data = shm.get_contents_as_numpy(
        shm_op0_handle, utils.triton_to_np_dtype(output.datatype),
        output.shape)
    else:
        print("OUTPUT is missing in the response.")
        sys.exit(1)
    print(output_data.shape)
    print(triton_client.get_system_shared_memory_status())

    save_image(output_data)

    triton_client.unregister_system_shared_memory()
    shm.destroy_shared_memory_region(shm_ip0_handle)
    shm.destroy_shared_memory_region(shm_op0_handle)
    

def load_image(url:str, color:str ="RGB") -> np.ndarray:
    folder_name, file_name = os.path.split(url)
    file_name_noext, file_ext = os.path.splitext(file_name)

    if os.path.isfile(url):
        image = pil_image.open(url)
        
    else:
        raise Exception("image no founded")
    
    image = np.asarray(image).astype(np.float32)
    image = convert_rgb_to_ycbcr(image)
    return image
def get_meta(url:str):

    return meta    

def convert_rgb_to_ycbcr(img, dim_order='hwc'):
    if dim_order == 'hwc':
        y = 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
        cb = 128. + (-37.945 * img[..., 0] - 74.494 * img[..., 1] + 112.439 * img[..., 2]) / 256.
        cr = 128. + (112.439 * img[..., 0] - 94.154 * img[..., 1] - 18.285 * img[..., 2]) / 256.
    else:
        y = 16. + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.
        cb = 128. + (-37.945 * img[0] - 74.494 * img[1] + 112.439 * img[2]) / 256.
        cr = 128. + (112.439 * img[0] - 94.154 * img[1] - 18.285 * img[2]) / 256.
    return np.array([y, cb, cr]).transpose([1, 2, 0])

def convert_ycbcr_to_rgb(img, dim_order='hwc'):
    if dim_order == 'hwc':
        r = 298.082 * img[..., 0] / 256. + 408.583 * img[..., 2] / 256. - 222.921
        g = 298.082 * img[..., 0] / 256. - 100.291 * img[..., 1] / 256. - 208.120 * img[..., 2] / 256. + 135.576
        b = 298.082 * img[..., 0] / 256. + 516.412 * img[..., 1] / 256. - 276.836
    else:
        r = 298.082 * img[0] / 256. + 408.583 * img[2] / 256. - 222.921
        g = 298.082 * img[0] / 256. - 100.291 * img[1] / 256. - 208.120 * img[2] / 256. + 135.576
        b = 298.082 * img[0] / 256. + 516.412 * img[1] / 256. - 276.836
    return np.array([r, g, b]).transpose([1, 2, 0])

def save_image(image : np.ndarray):
    image = convert_ycbcr_to_rgb(image).clip(0,255).astype(np.uint8)
    print(image.shape)
    image = pil_image.fromarray(image)
    image.save("temp.png")


if __name__ == "__main__":
    channel = 1
    model_name = "ECBSR_ensemble"
    scale = 2
    
    img = load_image("./images/GettyImages-1357529222.jpg","YCbCr")
    print(img)

    call_triton_inference(img,model_name,scale,channel)