import io, json, requests
KEY = '4f95e1b95c22a001687576e3c07fcd33'
out = {}
for k, im in res.items():
    b = io.BytesIO(); im.save(b, 'PNG')
    r = requests.post('https://orchestration.civitai.com/v2/consumer/blobs',
                      headers={'Authorization': 'Bearer ' + KEY, 'User-Agent': 'Mozilla/5.0', 'Content-Type': 'image/png'},
                      data=b.getvalue())
    j = r.json(); out[k] = j.get('id')
    print('UP', k, out[k])
print('UPJSON', json.dumps(out))
