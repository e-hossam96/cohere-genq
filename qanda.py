import json
import requests
from urllib.parse import urlencode


def main():
    domain = "http://127.0.0.1:5000"
    q = {"question": "What is data science?"}
    response = requests.get(domain, data=q)
    if response.status_code == 200:
        ans = json.loads(response.content)["answer"]
    else:
        print("Error {}".format(response.status_code))
        ans = None
    return ans


if __name__ == "__main__":
    print(main())
