#!/usr/bin/env bash
cd "$(dirname "$0")"
microk8s status --wait-ready
echo -n "waiting for pods init in namespace $1 ..."
while true; do
    echo -n "."
    sleep 3
    lines=$(kubectl get pod -n $1 |& grep -vw "Completed")
    nblines=$(wc -l <<< $lines)
    #echo "#######################################"
    #cat <<< $lines
    #cat <<< $nblines
    if [[ $nblines -eq 1 ]]; then
        continue
    fi
    if [ $(awk '{print $2}' <<< $lines | tail -n +2 | bc\
         | paste -s -d '+' | bc) -eq $((nblines-1)) ]; then
        break
    fi
done
echo ""