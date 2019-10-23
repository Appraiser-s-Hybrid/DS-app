import requests
url = 'http://localhost:5000/api'
r = requests.post(url,json={'bedroomcnt':4, 'bathroomcnt':4, 'regionidzip':96268})
print(r.json())