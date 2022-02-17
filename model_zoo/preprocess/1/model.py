import json
import numpy as np
import triton_python_backend_utils as pb_utils
import cv2

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = model_config = json.loads(args['model_config'])
    
    def preprocess(self, image : np.ndarray,channel) -> np.ndarray:
        if channel[0] == 1:
            cbcr = cv2.resize(image,dsize=(0,0),fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
            cbcr = cbcr[...,1:]
            image = image[...,0]
            image = np.expand_dims(image,axis=2)
            print(cbcr.shape,flush=True)
        else:
            cbcr = np.zeros((1,1,2))
        image = image.transpose([2, 0, 1])
        image /= 255.0
        image = np.expand_dims(image,axis=0)

        return image, cbcr

    def execute(self, requests):
        responses = []

        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            in_1 = pb_utils.get_input_tensor_by_name(request, "INPUT1")

            in_0 = in_0.as_numpy()
            in_1 = in_1.as_numpy()

            out_0, out_1 = self.preprocess(in_0,in_1)

            out_tensor_0 = pb_utils.Tensor("OUTPUT0", out_0.astype("float32"))
            out_tensor_1 = pb_utils.Tensor("OUTPUT1", out_1.astype("float32"))

            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0,out_tensor_1])

            responses.append(inference_response)

        return responses

    def finalize(self):
        print('Cleaning up...')