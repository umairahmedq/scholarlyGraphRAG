import os
import requests

# Directory containing your RDF files
rdf_dir = '/home/jovyan/data/wikidata/scholarlydatadb'

# Fuseki endpoint for data insertion
endpoint = 'http://localhost:3030/schdb/data'
headers = {'Content-Type': 'application/rdf+xml'}

# Loop through each RDF file in the directory
for filename in os.listdir(rdf_dir):
    if filename.endswith('.rdf'):
        filepath = os.path.join(rdf_dir, filename)
        print(f"Uploading file: {filepath}")
        with open(filepath, 'rb') as f:
            rdf_data = f.read()
        response = requests.post(endpoint, headers=headers, data=rdf_data)
        print(f"Status code: {response.status_code}")
        if response.ok:
            print("Upload successful.")
        else:
            print("Upload failed:", response.text)
