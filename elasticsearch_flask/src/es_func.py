# Implementation of Py-ElasticSearch

def return_query(txt_search, qtype = 1):
	# txt_search: "text" user is looking for 
	# qtype: This will implement different queries 
	# OUTPUT: query -- like JSON format
	
	# JUST for quick implementation, set qtype=1
	query = None
	
	# We will return this query everytime, this function needs to be populated
	if qtype == 1:
		query = {
			"query": {
				"match" : {"content" : txt_search}
			},
			"highlight": {
			    "pre_tags" : ["<mark>"],
				"post_tags" : ["</mark>"],
				"fields": {
					"content": {"fragment_size" : 500,
								"number_of_fragments": 1000}
				}
			}
		}
	elif qtype == 2: # offsets
		query = {
			"query": {
				"match_phrase" : {"content" : txt_search}
			},
			"highlight": {
				"pre_tags" : ["<b>"],
				"post_tags" : ["</b>"],
				"fields": {
					"content": {
						"fragment_size" : 100,
						"number_of_fragments": 1000, 
						"options": {
							"hit_source": "analyze",
							"return_offsets": "true"
						}
					}
				},
				"order": "score"
			}
		}
	elif qtype == 3:
		txt_search = " OR ".join(txt_search.split(','))

		query = {
			"query": {
				"query_string" : {"query": txt_search, "fields": ["content"]}
			},
			"highlight": {
				"fields": {
					"content": {"fragment_size" : 300, "number_of_fragments": 1000}
				}
			}
		}

	return query

def execute_search(es, txt_search, index, qtype, logger):
	# es: elasticsearch object
	# txt_search: "text" user is looking for 
	# index: index
	# qtype: Query type
	# OUTPUT: res -- response object from elasticsearch

	logger.debug('Search Started')
	#print('Search Started')

	# Generated Index
	generated_index = "!!YOUR index SIGNATURE HERE!!!"

	# Filtering fields
	filter_search = ['hits.hits._id', 'hits.hits._type', 'hits.total', 'hits.hits.highlight']
	
	# get the query
	query = return_query(txt_search, qtype)

	# Do the searching
	res = es.search(index=generated_index, doc_type = '_doc', size=10000, body = query, filter_path=filter_search)
	logger.debug("Search Complete.")
	#print("Search Complete.")
	#print(res)
	
	# return response
	return res

