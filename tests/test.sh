# usage
#     nohup bash ./test.sh & echo $! > pidfile
#

singularity exec container.img python bin/text.py -t 2 -RS 5 -i 2 > text.out &&\
    singularity exec container.img python bin/list.py -t 2 -RS 5 -i 2 > list.out && \
    singularity exec container.img python bin/logo.py -t 5 -RS 10 --biasOptimal -i 2 > logo.out && \
    singularity exec container.img python bin/demo3.py -t 2 --testingTimeout 2 -i 2 > demo.out
