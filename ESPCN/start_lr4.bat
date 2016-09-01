@echo start training ESPCN(5x3x3)
caffe train --solver=./ESPCN_solver.prototxt --snapshot=./snapshot/espcn_iter_5000000.solverstate
pause