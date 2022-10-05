# claim-5: clean
# 	./claim-5.sh
# 	./analyze.sh

# claim-5-viz:
# 	./analyze.sh

test-1:
	pypy3 bin/sasquatch.py -v -a 0 -i -t --iteration 0

test-2:
	pypy3 bin/sasquatch.py -v -a 1 -i -t --iteration 0 -r

clean:
	rm -rf artifact_out/*

.PHONY: claim-5 claim-5-viz clean test-1 test-2