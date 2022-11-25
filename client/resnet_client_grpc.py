from __future__ import print_function
import grpc
import requests
import numpy as np
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

IMAGE_URL = 'https://tensorflow.org/images/blogs/serving/cat.jpg'

tf.compat.v1.app.flags.DEFINE_string('server', 'localhost:8500',
                                     'PredictionService host:port')
tf.compat.v1.app.flags.DEFINE_string('image', './cat.jpg',
                                     'path to image in JPEG format')
FLAGS = tf.compat.v1.app.flags.FLAGS


def main(_):
  if FLAGS.image:
    with open(FLAGS.image, 'rb') as f:
      data = f.read()
  else:
    dl_request = requests.get(IMAGE_URL, stream=True)
    dl_request.raise_for_status()
    data = dl_request.content

  channel = grpc.insecure_channel(FLAGS.server)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
  
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'resnet'
  request.model_spec.signature_name = 'serving_default'
  request.inputs['image_bytes'].CopyFrom(
      tf.make_tensor_proto(data, shape=[1]))
  
  result = stub.Predict(request, 10.0)  # 10 secs timeout
  result = result.outputs['probabilities'].float_val
  print("class as  "+str(np.argmax(result)))


if __name__ == '__main__':
  tf.compat.v1.app.run()