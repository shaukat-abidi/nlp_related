# ELASTICSEARCH API

## Web Application

A simple [Flask](http://flask.pocoo.org/) web app for elasticsearch. [Flasgger](https://github.com/rochacbruno/flasgger) is used to generate [Swagger](https://swagger.io/) specification. Another Python options for Swagger documentation for Flask views is [Connexion](https://connexion.readthedocs.io/en/latest/) - Try it! 

It can be easily be deployed or run using Docker.

## Docker

Build docker image

	docker build -t es_textsearch .

Start container using the docker image you have built. Replace HOST_PORT with the port on the host machine you want to run the container at (I am running flask on port 5060)

	docker run --name es_textsearch_instance -p HOST_PORT:5060 -d es_textsearch

Once the docker container has been run you should be able to access it at (host should be reachable from remote machine - see public IP for example)

	http://host:HOST_PORT

Code for implementing and deploying Elasticsearch Swagger API:

*src/elasticsearch_service.py*:  establishes a connection to elasticsearch node

*src/es_connection.py*: returns an instance of elasticsearch object

*src/es_func.py*: contains queries that perform search on indexed documents

*search.yml*: yml file


# Running docker container

The following prerequisites must be met before running the above container

# Prerequisites

1. Pull docker images for Elasticsearch
> docker pull docker.elastic.co/elasticsearch/elasticsearch:6.3.2

2. Now follow instructions [here](https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html) to run Elasticsearch node(s) in Production mode

3. Index your documents and then:

> docker run --name es_textsearch_instance -p HOST_PORT:5060 -d es_textsearch
