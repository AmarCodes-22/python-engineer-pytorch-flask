import requests 

# https://your-heroku-app-name.herokuapp.com/predict
# http://localhost:5000/predict

# resp = requests.post("https://pytorch-flask-deployment-test.herokuapp.com/predict", files={'file': open('three.png', 'rb')})
resp = requests.post("http://localhost:5000/predict", files={'file': open('9.jpg', 'rb')})

# print(resp.text)
print(resp.text)
