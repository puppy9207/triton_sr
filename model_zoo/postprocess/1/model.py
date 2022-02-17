import json
import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = model_config = json.loads(args['model_config'])
    
    def postprocess(self, image : np.ndarray, cbcr : np.ndarray) -> np.ndarray:
        image = image.squeeze(0)
        image *= 255.0
        image = np.clip(image,0.0,255.0)
        image = image.transpose([1,2,0])
        if image.shape[2] == 1:
            image = np.array([image[...,0],cbcr[...,0],cbcr[...,1]])
        image = image.transpose([1,2,0])
        return image

    def execute(self, requests):
        responses = []

        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")

            in_0 = in_0.as_numpy()
            
            in_1 = pb_utils.get_input_tensor_by_name(request, "INPUT1")

            in_1 = in_1.as_numpy()

            out_0 = self.postprocess(in_0,in_1)

            out_tensor_0 = pb_utils.Tensor("OUTPUT0", out_0.astype("uint8"))

            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0])

            responses.append(inference_response)

        return responses

    def finalize(self):
        print('Cleaning up...')