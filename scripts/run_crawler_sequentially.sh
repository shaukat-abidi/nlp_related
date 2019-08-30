#!/bin/bash

# Vars
accepted=0
limit="5"

# Run the script that runs indexing sequentially
for i in {1,2,3,4,5,6}
do
	# Initializing vars
	retry="0"
	current_user="user_$i"
	echo "Running Crawler for: "$current_user
	sudo -u ubuntu /path_to_fscrawler/bin/fscrawler $current_user --loop 1 --restart >> /path_to_logs/log.txt 2>&1 &
	pid=$!
	echo 'pid for ' $current_user ':' $pid
	echo 'waiting for pid: ' $pid ' to get finished.'
	wait $pid
	exit_code=$?
	echo 'pid:' $pid ' finished with the exit code: ' $exit_code 'Sleeping for 2 seconds before continuing'
	sleep 2
	
	while [ "$exit_code" != "$accepted" ]
	do
		# increment counter
		retry=$[$retry + 1]
		echo "Re-running Crawler for: "$current_user "Retrying: " $retry
		sudo -u ubuntu /path_to_fscrawler/bin/fscrawler $current_user --loop 1 --restart >> /path_to_logs/log.txt 2>&1 &
		pid=$!
		echo 'pid for ' $current_user ':' $pid
		echo 'waiting for pid: ' $pid ' to get finished.'
		wait $pid
		exit_code=$?
		echo 'pid:' $pid ' finished with the exit code: ' $exit_code 'Sleeping for 1 seconds before continuing'
		sleep 1
		if [ "$retry" = "$limit" ];then
			echo 'breaking while loop as number of retries is exhausted - Retried: ' $retry 'times'
			break
		fi
	done
done
