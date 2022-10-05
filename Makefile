claim-5:
	./claim-5.sh
	./analyze.sh

claim-5-viz:
	./analyze.sh

clean:
	rm -rf artifact_out/*

.PHONY: claim-5 claim-5-viz clean