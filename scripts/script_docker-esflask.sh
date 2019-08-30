#!/bin/bash

# NOT A GOOD WAY TO RUN THE CONTAINER

# Running docker in background
docker run --net="host" --name es_textsearch_instance -p 5060:5060 es_textsearch &>/path_to_logs/es_textsearch_output.txt &
pid=$!
echo 'pid_for_docker_container disappearing soon:' $pid
sleep 5

#checking if docker container started successfully
status_code="$(docker inspect es_textsearch_instance --format='{{.State.ExitCode}}')"
echo 'status_code for es_textsearch_instance: ' $status_code
accepted=0
limit="14"
retry="0"

while [ "$status_code" != "$accepted" ]
do

	echo 'Docker container initialization failed. Retrying now ---' $retry

	# remove the old container
	docker rm -f es_textsearch_instance
	sleep 3

	# rerun the new container
	docker run --net="host" --name es_textsearch_instance -p 5060:5060 es_textsearch &>/path_to_logs/es_textsearch_output.txt &
	pid=$!
	echo 'Rerunning container with pid:' $pid
	sleep 5

	#checking if docker container started successfully
	status_code="$(docker inspect es_textsearch_instance --format='{{.State.ExitCode}}')"
	echo 'status_code for es_textsearch_instance: ' $status_code

	if [ "$retry" = "$limit" ];then
		echo 'breaking while loop'
		break
	fi

	# increment counter
	retry=$[$retry + 1]

done

if [ "$retry" -ge "$limit" ];then
	echo 'Docker container failed to initialize, sending exit status 1'
	exit 1
else
	echo 'Docker container running, sending exit status 0'
exit 0
fi
