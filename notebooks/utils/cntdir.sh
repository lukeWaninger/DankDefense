#!/bin/bash

while true; do
        printf "\033c"	
	s=$(ls $1 -x -s -h -1 -b | grep $2)
	echo "$s"

	if [ "$3" != "" ]; then
		sleep $3
	else
		sleep 5s
	fi
done
