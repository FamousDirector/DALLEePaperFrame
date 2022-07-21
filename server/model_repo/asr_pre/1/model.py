import sys

import numpy as np
import json
import scipy.signal as sps

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils
import os


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    @staticmethod
    def normalize(x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return np.squeeze((x - mean) / np.sqrt(var + 1e-5))

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "preprocessed_audio")

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])

        # preprocessing values
        self.new_rate = 16000

    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        output0_dtype = self.output0_dtype

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            raw_audio_array = pb_utils.get_input_tensor_by_name(request, "raw_audio_data").as_numpy()[0]
            sampling_rate = pb_utils.get_input_tensor_by_name(request, "sampling_rate").as_numpy()

            # preprocessing
            num_of_channels = raw_audio_array.shape[-1]
            raw_audio_array = raw_audio_array.sum(axis=-1) / num_of_channels  # multi channel (stereo) to mono

            sample_length = int(len(raw_audio_array) * float(self.new_rate) / sampling_rate)

            if sample_length == 0:
                sample_length = len(raw_audio_array)
                print("Error: sample_length is 0, defaulting to original length")
            elif sample_length < 0:
                sample_length = len(raw_audio_array)
                print("Error: sample_length is negative, defaulting to original length")
            elif sample_length > len(raw_audio_array) * 10:
                sample_length = len(raw_audio_array)
                print("Error: sample_length is greater than 10 times original length, defaulting to original length")

            new_data = sps.resample(raw_audio_array, sample_length)
            speech = np.array(new_data, dtype=np.float32)

            speech = self.normalize(speech)[None]

            out_tensor_0 = pb_utils.Tensor("preprocessed_audio",
                                           speech.astype(output0_dtype))

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occured"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0])
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
