import json
import requests
import time

for i in range(90):
    QRN_URL = "https://api.quantumnumbers.anu.edu.au/"
    QRN_KEY = "WMw8APwUGW6a4bSreglGn22RFSDTWCtL8EFNiPyK"  # replace with your secret API-KEY

    DTYPE = "hex16"  # uint8, uint16, hex8, hex16
    LENGTH = 1024 # between 1--1024
    BLOCKSIZE = 10  # between 1--10. Only needed for hex8 and hex16

    params = {"length": LENGTH, "type": DTYPE, "size": BLOCKSIZE}
    headers = {"x-api-key": QRN_KEY}

    response = requests.get(QRN_URL, headers=headers, params=params)

    if response.status_code == 200:
        js = response.json()
        if js["success"] == True:
            f = open("numbers.bin", "ab")
            f.write(bytes.fromhex("".join(js["data"])))
            print("added_data")
        else:
            print(js["message"])

    else:
        print(f"Got an unexpected status-code: {response.status_code}")
        print(response.text)