#!/usr/bin/env bash
cd "$(dirname "$0")"
microk8s status --wait-ready
while true; do
    nbline=$(kubectl get pod -A  | wc -l)
    if [ $nbline -lt 3 ]; then
        sleep 5
        continue
    fi
    if [ $(kubectl get pod -A | awk\
         '{print $3}' | tail -n +2 | bc | paste\
          -s -d '+' | bc) -eq $((nbline-1)) ]; then
        break
    fi
    sleep 5
done
