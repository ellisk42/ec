#!/bin/bash
# syncs model results only

rsync -avz lyt@openmind7.mit.edu:/om/user/lyt/ec/jobs .
rsync -avz lyt@openmind7.mit.edu:/om/user/lyt/ec/experimentOutputs .

