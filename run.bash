printf "\n[ $(date) ]\n\n" >> log.txt;
if (( $# == 3 )) ; then
    python2 model.py "$@" | tee -a log.txt;
else
    python2 model.py 0.001 0.8 3 | tee -a log.txt;
fi