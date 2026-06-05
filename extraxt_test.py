import requests

url = "https://www.feynmanlectures.caltech.edu/I_01.html"

response = requests.get(url)

print("Status:", response.status_code)

print("\nFirst 1000 characters:\n")
print(response.text[:1000])