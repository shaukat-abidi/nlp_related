# https://elasticsearch-py.readthedocs.io

import requests
from elasticsearch import Elasticsearch
import json

# ES is running on host
res = requests.get('http://127.0.0.1:9200')
print(res) # Response 200 is required

# connect to elastic search on host (localhost)
es = Elasticsearch([{'host': '127.0.0.1', 'port': 9200}])

# get all indices 
print(es.indices.get_alias("*"))

# Search for text
txt_search = 'combination of events'
filter_path = None #['hits.hits._id', 'hits.hits._type']
query = {
    "query": {
        "match" : { 
        "content" : txt_search 
        }
    }
}
generated_index = 'your_index'
res = es.search(index = generated_index, doc_type = '_doc', body = query, filter_path=filter_path)
print(res)

# Highlighting
txt_search = 'date'
filter_path = ['hits.hits._id', 'hits.hits._type', 'hits.total', 'hits.hits.highlight']

query = {
    "query": {
        "match" : {"content" : txt_search}
    },
    "highlight": {
        "fields": {
            "content": {"fragment_size" : 100, 
                        'number_of_fragments': 20}
        }
    }
}
res = es.search(index=generated_index, doc_type = '_doc', body = query, filter_path=filter_path)

# Deleting Index
es.indices.delete(index=generated_index)
