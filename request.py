import requests
import json
url = 'http://localhost:5000/'
data = {'bedroomcnt': 4
      , 'bathroomcnt': 4
      , 'regionidzip': 96268}
data = json.dumps(data)