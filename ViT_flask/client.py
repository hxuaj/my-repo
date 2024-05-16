# -*- coding: utf-8 -*-
import requests
import time
import json
import base64


url = 'http://0.0.0.0:5000/predict'
# url = 'http://172.17.0.3:5000/predict'

# image_files = 'lemon_pic.jpg'
image_names = ['lemon_pic.jpg', 'test1.jpg', 'test2.jpg']
# image_names = ['lemon_pic.jpg', 'lemon_pic.jpg', 'lemon_pic.jpg']
# image_names = ['test1.jpg'] * 5

inputs = []
for i, image_name in enumerate(image_names):
    with open(image_name, 'rb') as image_files:
        bytes_inputs = image_files.read()
    b64str = base64.urlsafe_b64encode(bytes_inputs).decode("utf-8")
    inputs.append({"filename": image_name, "image": b64str})

data = json.dumps({"inputs": inputs})
headers = {'Content-Type': 'application/json'}

# send images through file like object
# files = [('file', open(file, 'rb')) for file in image_paths]
# files = {}
# for idx, file in enumerate(image_paths, start=1):
#     files[f'file{idx}'] = open(file, 'rb')

tic = time.perf_counter()
# response = requests.post(url, files=files)
response = requests.post(url, data=data, headers=headers)
toc = time.perf_counter()
print("request time: ", toc - tic)
# print(response.text)

decoded_response = bytes(response.text, 'utf-8').decode('unicode_escape')
# Print the decoded response
print(decoded_response)
