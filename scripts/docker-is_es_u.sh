#!/bin/bash

# Check if docker container is up and running
status_code="$(docker inspect elasticsearch_instance --format='{{.State.ExitCode}}')"

# Vars
accepted=0
limit="14"
retry="0"
broken="0"

echo 'status_code for es_textsearch_instance: ' $status_code

while [ "$status_code" != "$accepted" ];do
	echo 'Docker Container not up yet...Sleeping for 5s'
	sleep 5
	status_code="$(docker inspect elasticsearch_instance --format='{{.State.ExitCode}}')"
	echo 'status_code for es_textsearch_instance: ' $status_code '--- Retried: ' $retry
	if [ "$retry" = "$limit" ];then
		echo 'breaking while loop'
		broken="1"
		break
	fi

	# increment counter
	retry=$[$retry + 1]
done

if [ "$broken" -eq 1 ];then
	echo 'Reached maximum number of retries. Exiting with code 1'
	exit 1
	else
		echo 'Container - es_textsearch_instance -  is up now'
fi

# Sleep for 5 seconds so that container is up properly
sleep 5

# Run the script that runs indexing sequentially
/path_to_scripts/run_crawler_sequentially.sh

