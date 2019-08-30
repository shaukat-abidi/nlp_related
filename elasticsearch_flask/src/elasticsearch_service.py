from flask import Flask, redirect, request
from flask.json import jsonify
from flasgger import Swagger
from flasgger import swag_from
import requests
import es_func
import es_connection as con
import argparse
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)

template = {
	"swagger": "2.0",
	"info":
	{
		"title": "Elastic Search Demo",
		"description": "API endpoint for conducting free-text search using ElasticSearch, output is returned in JSON.",
		"version": "0.0.1",
		"contact":
		{
			"name": "Preferred Name here",
			"email": "Your email here",
			"url": "Your URL here",
		}
	},
	"tags":
	[
		{
			"name": "Elasticsearch",
			"description": "Conducting search in ElasticSearch"
		}
	],
}

swagger_config = {
	"headers": [
	],
	"specs": [
		{
			"endpoint": 'apispec_1',
			"route": '/v2/api-docs',
			"rule_filter": lambda rule: True,  # all in
			"model_filter": lambda tag: True,  # all in
		}
	],
	"static_url_path": "/flasgger_static",
	"swagger_ui": True,
	"specs_route": "/docs/",
}

app = Flask(__name__)
app.config['SWAGGER'] = {
	'title': 'Elasticsearch Demo', # webpage title
	'uiversion': 3,
	"docExpansion": "true"
}
Swagger(app, template=template, config=swagger_config)

@app.route('/')
def root():
	return redirect("docs", code=302)

@app.route('/api/searchText', methods=['POST'])
@swag_from('search.yml')
def search_endpoint():
	txt_search = request.args.get("text")
	# print(type(request_text))
	# print(request_text)
	
	# Setting parameters 
	qtype = 1
	index = request.args["Indices"] #3
	logger.info('Indices: {}'.format(request.args["Indices"]))

	output_returned = es_func.execute_search(es, txt_search, index, qtype, logger)
	return jsonify(output_returned)

def elasticsearch_service(_host, _port):
	#app.run(host='0.0.0.0')
	app.run(host=_host, port=_port)

if __name__ == "__main__":
	global es
	
	# Parse the arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--host_es', type=str, default='127.0.0.1', help='Host where ES is running')
	parser.add_argument('--host_flask', type=str, default='0.0.0.0', help='FLASK server host')
	parser.add_argument('--port_es', type=int, default=9200, help='ES port')
	parser.add_argument('--port_flask', type=int, default=5060, help='FLASK port')
	args = parser.parse_args()

	# ES is running on the following host:port
	_hostname = args.host_es
	_port = args.port_es 
	
	connect_obj = con.connection(_host = _hostname, _port = _port)
	es = connect_obj.connect(logger)

	elasticsearch_service(args.host_flask, args.port_flask)
