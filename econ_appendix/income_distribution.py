#%%
import io, zipfile, requests, pandas as pd
from pathlib import Path

def download_pums(state_abbr, year=2023):
    url = f"https://www2.census.gov/programs-surveys/acs/data/pums/{year}/1-Year/csv_h{state_abbr.lower()}.zip"
    buf = io.BytesIO(requests.get(url, timeout=60, verify=False).content)
    with zipfile.ZipFile(buf) as z:
        # Housing file always begins psam_h
        csv_name = [f for f in z.namelist() if f.startswith("psam_h")][0]
        return pd.read_csv(z.open(csv_name), low_memory=False,
                           usecols=["SERIALNO", "HINCP", "ADJINC", "WGTP"])

# Example: pull just CA to keep this demo light
df = download_pums("ca")

# Convert to 2023 dollars
df["HINCP_adj"] = df["HINCP"] * df["ADJINC"] / 1_000_000
df = df.loc[df["HINCP_adj"] > 0]          # drop loss / missing
df = df.loc[df["WGTP"] > 0]               # keep valid weights

#%%
