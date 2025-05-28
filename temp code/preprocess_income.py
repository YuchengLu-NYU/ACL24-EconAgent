#%%

import io
import zipfile
import requests
import pandas as pd
from pathlib import Path

#%%
# 1. List of all 50 states + DC + PR (2-letter USPS codes, lowercase)

STATE_ABBRS = [
    "ak", "al", "ar", "az", "ca", "co", "ct", "dc", "de", "fl", "ga", "hi",
    "ia", "id", "il", "in", "ks", "ky", "la", "ma", "md", "me", "mi", "mn",
    "mo", "ms", "mt", "nc", "nd", "ne", "nh", "nj", "nm", "nv", "ny", "oh",
    "ok", "or", "pa", "pr", "ri", "sc", "sd", "tn", "tx", "ut", "va", "vt",
    "wa", "wi", "wv", "wy"
]

#STATE_ABBRS = ['ca']

def download_and_process_pums(state_abbr, year=2023, save_dir="pums_raw_zips"):
    # ensure save directory exists
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # download URL & local filename
    url = f"https://www2.census.gov/programs-surveys/acs/data/pums/{year}/1-Year/csv_h{state_abbr}.zip"
    zip_path = Path(save_dir) / f"csv_h{state_abbr}.zip"

    # 2. Download & save ZIP
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    zip_path.write_bytes(resp.content)

    # 3. Read the "psam_h…" CSV inside the ZIP
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        csv_name = next(f for f in z.namelist() if f.startswith("psam_h"))
        df = pd.read_csv(z.open(csv_name), low_memory=False,
                         usecols=["SERIALNO", "HINCP", "ADJINC", "WGTP"])

    # 4. Add state code, adjust income, and filter
    df["STATE"]     = state_abbr.upper()
    df["HINCP_adj"] = df["HINCP"] * df["ADJINC"] / 1_000_000
    df = df[(df["HINCP_adj"] > 0) & (df["WGTP"] > 0)]
    return df
#%%
# 5. Loop over all states, collect into a list
all_dfs = []
for abbr in STATE_ABBRS:
    print(f"→ {abbr.upper()} ", end="", flush=True)
    try:
        df_state = download_and_process_pums(abbr)
        all_dfs.append(df_state)
        print("Downloaded")
    except Exception as e:
        print(f"Error: {e}")


# 6. Concatenate and report
acs_national = pd.concat(all_dfs, ignore_index=True)
print(f"\nTotal rows after cleaning: {len(acs_national):,}")

# 7. Save to CSV
output_csv = "acs_national_processed.csv"
acs_national.to_csv(output_csv, index=False)
print(f"Processed DataFrame saved to: {output_csv}")

# %%
