import requests

r = requests.get("https://huggingface.co/api/models", params={
    "search": "facebook/dinov3-vith",
    "library": "transformers",
    "sort": "downloads",
    "limit": 10
})

for m in r.json():
    print(m["id"])