import requests
from tqdm import tqdm

NWS_ARCHIVE = "https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py"
DISCUSSION_DIR = "discussions"

start_date = "2026-04-01T00:00Z"
end_date = "2026-04-13T23:59Z"
request_params = {
    "pil": "AFDMTR",
    "fmt": "zip",
    "sdate": start_date, 
    "edate": end_date,
    "limit": 9999
}

with requests.get(NWS_ARCHIVE, params=request_params, stream=True) as response:
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    with open(f"{DISCUSSION_DIR}/afdmtr-{start_date}-{end_date}.zip", "wb") as f, \
        tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
