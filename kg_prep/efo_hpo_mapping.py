import requests
import csv
import time

API_KEY = "d5d51e88-b027-414f-9b01-caca9e4f88b5"  # Replace with your actual key
OUTPUT_FILE = "efo_hpo_mappings.csv"

base_url = "https://data.bioontology.org/mappings"
params = {
    "ontologies": "EFO,HP",
    "apikey": API_KEY,
    "pagesize": 100,  # max allowed is 500, but 100 is safer
    "page": 1
}

all_mappings = []

print("starting download.")

while True:
    print(f"Fetching page {params['page']}...")
    response = requests.get(base_url, params=params)
    
    if response.status_code != 200:
        print("Error:", response.status_code, response.text)
        break
    
    data = response.json()
    collection = data.get("collection", [])
    
    if not collection:
        print("No more data.")
        break

    for mapping in collection:
        classes = mapping.get("classes", [])
        if len(classes) == 2:
            # extract ontology names and IDs
            ids = [cls["@id"].split("/")[-1] for cls in classes]
            ontologies = [cls["links"]["ontology"].split("/")[-1] for cls in classes]
            
            # order: (EFO, HPO)
            if ontologies[0] == "EFO" and ontologies[1] == "HP":
                efo_id, hpo_id = ids[0], ids[1]
            elif ontologies[1] == "EFO" and ontologies[0] == "HP":
                efo_id, hpo_id = ids[1], ids[0]
            else:
                continue  # skip mappings that aren't EFO ↔ HPO

            all_mappings.append((efo_id, hpo_id))

    # pagination
    if "nextPage" in data.get("links", {}):
        params["page"] += 1
        time.sleep(0.5)  # polite delay
    else:
        break

print(f"got {len(all_mappings)} mappings. writing.")

# write to output
with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["EFO_ID", "HPO_ID"])
    writer.writerows(all_mappings)

print(f"done at {OUTPUT_FILE}")
