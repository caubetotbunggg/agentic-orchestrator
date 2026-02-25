import requests

with open("temp_image.png", "wb") as f:
    f.write(b"fake_png_data")

files = {
    'file': ('test.png', open('temp_image.png', 'rb'), 'image/png')
}
data = {
    'prompt': 'Can you extract the layout and then summarize what types of regions you found?',
    'src_lang': 'eng_Latn',
    'tgt_lang': 'vie_Latn'
}

try:
    response = requests.post("http://127.0.0.1:8000/api/v1/agent", files=files, data=data)
    print("Status:", response.status_code)
    import pprint
    pprint.pprint(response.json())
except Exception as e:
    print(f"Error: {e}")
