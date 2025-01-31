python train_lequa.py -t gmnet -n gmnet -f nofe -p /media/nas/olayap/env_olaya/Doctorado/gmnet/experiments/parameters/gmnet.json -b LeQuaBagGenerator -s -d T2 -c cuda:0
python train_lequa.py -t histnet -n histnet -f rff -p /media/nas/olayap/env_olaya/Doctorado/gmnet/experiments/parameters/histnet.json -b LeQuaBagGenerator -s -d T2 -c cuda:0
python train_lequa.py -t dqn_med -n deepsets -f rff -p /media/nas/olayap/env_olaya/Doctorado/gmnet/experiments/parameters/dqn_median.json -b LeQuaBagGenerator -s -d T2 -c cuda:0
python train_lequa.py -t dqn_max -n deepsets -f rff -p /media/nas/olayap/env_olaya/Doctorado/gmnet/experiments/parameters/dqn_max.json -b LeQuaBagGenerator -s -d T2 -c cuda:0
python train_lequa.py -t dqn_avg -n deepsets -f rff -p /media/nas/olayap/env_olaya/Doctorado/gmnet/experiments/parameters/dqn_avg.json -b LeQuaBagGenerator -s -d T2 -c cuda:0
