import numpy as np
import imageio
import cv2
import sys
from functools import partial
import os
import PIL.Image as pil_image

import tritonclient.grpc as grpcclient
from tritonclient import utils
from tritonclient.utils import InferenceServerException
import tritonclient.utils.shared_memory as shm

import queue

import natsort 


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()




# Callback function used for async_stream_infer()
def completion_callback(user_data, result, error):
    # passing error raise and handling out
    user_data._completed_requests.put((result, error))

def sharpning(image):
    strong = 13
    image = np.array(image).astype(np.uint8)
    sharp_filter = np.array([[-1, -1, -1, -1, -1],
                             [-1, 2, 2, 2, -1],
                             [-1, 2, strong, 2, -1],
                             [-1, 2, 2, 2, -1],
                             [-1, -1, -1, -1, -1]]) / strong
    image = cv2.filter2D(image, -1, sharp_filter)
    image = pil_image.fromarray(image)
    return image

def convert_rgb_to_y(img, dim_order='hwc'):
    if dim_order == 'hwc':
        return 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
    else:
        return 16. + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.

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

def preprocess(img):
    """
    Pre-process an image to meet the size, type and format
    requirements specified by the parameters.
    """
    if not isinstance(img, np.ndarray):
        img = np.asarray(img)
    img = img.astype(np.float32)
    # img_copy = img.copy()
    ycbcr = convert_rgb_to_ycbcr(img)
    # ycbcr = cv2.resize(ycbcr,(0,0),fx=2,fy=2)
    lr = ycbcr[..., 0]
    lr = np.expand_dims(lr,axis=0)
    return lr

def ycbcr_process(img):
    """
    Pre-process an image to meet the size, type and format
    requirements specified by the parameters.
    """
    if not isinstance(img, np.ndarray):
        img = np.asarray(img)
    img = img.astype(np.float32)
    # img_copy = img.copy()
    ycbcr = convert_rgb_to_ycbcr(img)
    ycbcr = cv2.resize(ycbcr,(0,0),fx=2,fy=2)
    return ycbcr

def postprocess(results,ycbcr):
    """
    Post-process results to show classifications.
    """
    print(ycbcr.shape)
    results = results.squeeze(0).squeeze(0)

    print(results.shape)
    results = np.array([results, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    results = np.clip(convert_ycbcr_to_rgb(results), 0.0, 255.0).astype(np.uint8)
    results = np.ascontiguousarray(results, dtype=np.uint8)

    return results


def requestGenerator(
    batched_image_data,
    input_name,
    output_name,
    dtype,
    sent_count,
):

    input_byte_size = batched_image_data.size * batched_image_data.itemsize
    output_byte_size = input_byte_size * (scale ** 2 + 1)

    shm_op0_handle.append(
        shm.create_shared_memory_region(
            f"output0_{sent_count}_data",
            f"/output0_{sent_count}_data",
            output_byte_size,
        )
    )

    # Register Output0 and Output1 shared memory with Triton Server
    triton_client.register_system_shared_memory(
        f"output0_{sent_count}_data", f"/output0_{sent_count}_data", output_byte_size
    )

    # Create Input0 and Input1 in Shared Memory and store shared memory handles
    shm_ip0_handle.append(
        shm.create_shared_memory_region(
            f"input0_{sent_count}_data", f"/input0_{sent_count}_data", input_byte_size
        )
    )

    # Register Input0 and Input1 shared memory with Triton Server
    triton_client.register_system_shared_memory(
        f"input0_{sent_count}_data", f"/input0_{sent_count}_data", input_byte_size
    )

    # Put input data values into shared memory
    shm.set_shared_memory_region(shm_ip0_handle[sent_count], [batched_image_data])

    # Set the input data
    inputs = []
    inputs.append(grpcclient.InferInput(input_name, batched_image_data.shape, dtype))
    inputs[0].set_shared_memory(f"input0_{sent_count}_data", input_byte_size)

    outputs = []
    outputs.append(grpcclient.InferRequestedOutput(output_name))
    outputs[0].set_shared_memory(f"output0_{sent_count}_data", output_byte_size)

    yield inputs, outputs


if __name__ == "__main__":
    model_name = "ECBSR"
    model_version = "1"  # default
    batch_size = 1  # default
    image_filename = "images"
    scale = 2

    url = "localhost:8001"

    input_name = "input"
    output_name = "output"
    dtype = "FP32"

    # Create gRPC client for communicating with the server
    try:
        triton_client = grpcclient.InferenceServerClient(url=url, verbose=False)
    except Exception as e:
        print("client creation failed: " + str(e))
        sys.exit(1)
    
    triton_client.unregister_system_shared_memory()
    # 이미지 경로를 이미지 데이터 리스트에 넣는 과정
    filenames = []
    if os.path.isdir(image_filename):
        filenames = [
            os.path.join(image_filename, f)
            for f in os.listdir(image_filename)
            if os.path.isfile(os.path.join(image_filename, f))
        ]
    else:
        filenames = [image_filename]
    # Preprocess the images into input data according to model
    filenames = natsort.natsorted(filenames)
    cnt = 0
    s_cnt = 0
    for p in range((len(filenames)//100)+1):
        image_data = []
        ycbcr_data = []
        for k in range(100):
            if cnt>=len(filenames):
                break
            image_data.append(preprocess(cv2.imread(filenames[cnt])))
            ycbcr_data.append(ycbcr_process(cv2.imread(filenames[cnt])))
            cnt += 1

        
            # image_data = [
            #     preprocess(cv2.imread(filename))
            #     for filename in filenames
            # ]
            # ycbcr_data = [
            #     ycbcr_process(cv2.imread(filename))
            #     for filename in filenames
            # ]

        user_data = UserData()

        triton_client.start_stream(partial(completion_callback, user_data))

        shm_op0_handle = []
        shm_ip0_handle = []

        sent_count = 0
        for data in image_data:
            batched_image_data = np.expand_dims(data, axis=0)
            print(batched_image_data.shape)
            # Send request
            try:
                for inputs, outputs in requestGenerator(
                    batched_image_data,
                    input_name,
                    output_name,
                    dtype,
                    sent_count,
                ):
                    sent_count += 1

                    triton_client.async_stream_infer(
                        model_name,
                        inputs,
                        outputs=outputs,
                    )

            except InferenceServerException as e:
                print("inference failed: " + str(e))
                triton_client.stop_stream()
                sys.exit(1)
        triton_client.stop_stream()
        
        for i in range(sent_count):
            results, error = user_data._completed_requests.get()

            if error is not None:
                print(f"{filenames[i]} inference failed: {error}")

                triton_client.unregister_system_shared_memory(f"input0_{i}_data")
                triton_client.unregister_system_shared_memory(f"output0_{i}_data")

                shm.destroy_shared_memory_region(shm_ip0_handle[i])
                shm.destroy_shared_memory_region(shm_op0_handle[i])
                continue

            output0 = results.get_output(output_name)

            output0_data = shm.get_contents_as_numpy(
                shm_op0_handle[i],
                utils.triton_to_np_dtype(output0.datatype),
                output0.shape,
            )
            
            folder_name, file_name = os.path.split(filenames[s_cnt])
            file_name_noext, file_ext = os.path.splitext(file_name)
            file_sr_path = os.path.join(
                f"{folder_name}_sr", f"{file_name_noext}_{model_name}{file_ext}"
            )

            if os.path.isfile(f"./compare/{file_name_noext}.jpg"):
                compare_img = pil_image.open(f"./compare/{file_name_noext}.jpg")
                origin_img = pil_image.open(f"./images/{file_name_noext}.jpg")
            w,h = origin_img.size
            icc = compare_img.info.get('icc_profile')
            exif = compare_img.info['exif']

            sr = postprocess(output0_data,ycbcr_data[i])
            sr = cv2.resize(sr,dsize=(w,h),interpolation=cv2.INTER_AREA)
            sr = cv2.cvtColor(sr,cv2.COLOR_BGR2RGB)
            sr = pil_image.fromarray(sr)
            # sr = sharpning(sr)
            os.makedirs(f"{folder_name}_sr", exist_ok=True)
            # cv2.imwrite(file_sr_path, postprocess(output0_data,ycbcr_data[i]))

            sr.save(file_sr_path, format='JPEG', exif=exif,icc_profile=icc, subsampling=0,quality=65)

            triton_client.unregister_system_shared_memory(f"input0_{i}_data")
            triton_client.unregister_system_shared_memory(f"output0_{i}_data")

            shm.destroy_shared_memory_region(shm_ip0_handle[i])
            shm.destroy_shared_memory_region(shm_op0_handle[i])
            s_cnt += 1

        print("done")
