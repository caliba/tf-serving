import numpy as np
from PIL import Image
import json
import requests
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import grpc
import numpy as np


class PredictModelGrpc(object):
    def __init__(self, model_name, input_name, output_name, socket='0.0.0.0:8500'):
        self.socket = socket
        self.model_name = model_name
        self.input_name = input_name
        self.output_name = output_name
        self.request, self.stub = self.__get_request()

    def __get_request(self):
        channel = grpc.insecure_channel(self.socket, options=[('grpc.max_send_message_length', 1024 * 1024 * 1024),
                                                              ('grpc.max_receive_message_length',
                                                               1024 * 1024 * 1024)]) 
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        request = predict_pb2.PredictRequest()

        request.model_spec.name = self.model_name
        request.model_spec.signature_name = "serving_default"

        return request, stub

    def inference(self, frames):
        self.request.inputs[self.input_name].CopyFrom(tf.compat.v1.make_tensor_proto(frames, dtype=tf.float32))
        result = self.stub.Predict.future(self.request, 10.0)
        res = tf.make_ndarray(result.result().outputs[self.output_name])[0]
        return res


class PredictModelRESTAPI(object):
    def __init__(self, model_name, input_name, output_name, socket='localhost:8500'):
        self.socket = socket
        self.model_name = model_name
        self.input_name = input_name
        self.output_name = output_name
        self.url = self._get_url()

    def _get_url(self):
        url = "http://{}/v1/models/{}:predict".format(self.socket, self.model_name)
        # print(url)
        return url

    def inference(self, data):
        payload = {
            "instances": [{self.input_name: data.tolist()}]
        }
        
        r = requests.post(self.url, json=payload)
        pred = json.loads(r.content.decode('utf-8'))
        pred = np.array(pred['predictions'][0])
        return pred


image_path = './test_images/test_img.png'
img = Image.open(image_path)
img = np.array(img) / 255.0


print("--- Different ways to access the same models ---")
model = PredictModelRESTAPI(model_name='mnist_c', input_name='flatten_input', output_name='dense_1', socket='localhost:8501')
res = model.inference(img)

model = PredictModelGrpc(model_name='mnist_c', input_name='flatten_input', output_name='dense_1', socket='0.0.0.0:8500')
res = model.inference(img)


