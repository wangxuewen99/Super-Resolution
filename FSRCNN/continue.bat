@echo start training fsrcnn
caffe train --solver=./fsrcnn_solver.prototxt --snapshot=./snapshot/snapshot_iter_2182500.solverstate
pause