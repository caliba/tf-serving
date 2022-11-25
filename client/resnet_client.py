from __future__ import print_function

import base64
import io
import json
import numpy as np
import requests

SERVER_URL = 'http://localhost:8501/v1/models/resnet:predict'
IMAGE_URL = './test_images/cat.jpg'


def main():
  with open('IMAGE_URL', 'rb') as f:
      data = f.read()
      
  jpeg_bytes = base64.b64encode(data).decode('utf-8')
  predict_request = '{"instances" : [{"b64": "%s"}]}' % jpeg_bytes

  for _ in range(3):
    response = requests.post(SERVER_URL, data=predict_request)
    response.raise_for_status()

  total_time = 0
  num_requests = 10
  for _ in range(num_requests):
    response = requests.post(SERVER_URL, data=predict_request)
    response.raise_for_status()
    total_time += response.elapsed.total_seconds()
    prediction = response.json()['predictions'][0]

  print('Prediction class: {}, avg latency: {} ms'.format(
      prediction['classes'], (total_time*1000)/num_requests))


if __name__ == '__main__':
  main()